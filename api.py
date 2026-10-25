"""
South India Open Data Integration Engine - REST API
FastAPI backend exposing data catalog, data quality profiling, resolver operations,
dataset integration graph, EV coverage explorer, interactive review queue, and reports.
Strictly parameterised DuckDB SELECT queries; no secrets in logs or responses.
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO
from datetime import datetime, timezone

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import DATA_DIR, SEED, TARGET_STATES, WORK_DIR
from ev import get_data_limitations_notice, get_ev_state_comparison, get_highway_corridors
from joins import build_dataset_graph_json
from reports import generate_data_quality_report, generate_release_zip, generate_suspected_error_report
from resolver import DistrictResolver, compute_similarity_features, load_canonical_reference_from_db

logger = logging.getLogger("api")

app = FastAPI(
    title="South India Open Data Integration Engine",
    description="ML-driven data standardisation and integration engine for South Indian state open datasets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = WORK_DIR / "engine.duckdb"
FRONTEND_FILE = Path(__file__).resolve().parent / "frontend" / "index.html"
LABELS_DIR = DATA_DIR / "labels"
QUEUE_FILE = LABELS_DIR / "labelling_queue.csv"
HAND_LABELS_FILE = LABELS_DIR / "hand_labels.csv"
EVAL_RESULTS_FILE = WORK_DIR / "results.json"
FEEDBACK_FILE = LABELS_DIR / "user_corrections.csv"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def get_db_connection() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not initialized. Run pipeline.py first.")
    return duckdb.connect(str(DB_PATH), read_only=True)


# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================
class SystemStatus(BaseModel):
    data_mode: str
    database_connected: bool
    target_states: List[str]
    banner: str
    table_counts: Dict[str, int]
    labels_count: int
    queue_pending_count: int


class ResolveRequest(BaseModel):
    state: str = Field(..., description="Target South Indian state")
    source_name: str = Field(..., min_length=1, max_length=200, description="Raw district string to match")
    threshold: float = Field(0.85, ge=0.0, le=1.0, description="Similarity threshold for automatic-link decisions")


class ResolveResponse(BaseModel):
    state: str
    source_name: str
    baseline_exact: Optional[str]
    baseline_exact_score: float
    baseline_tfidf: Optional[str]
    baseline_tfidf_score: float
    baseline_rapidfuzz: Optional[str]
    baseline_rapidfuzz_score: float
    top_candidates: List[Dict[str, Any]]
    model_prediction: Optional[str]
    model_confidence: float
    decision_band: str


class CorrectionSubmission(BaseModel):
    state: str = Field(..., min_length=1, max_length=100)
    source_name: str = Field(..., min_length=1, max_length=200)
    chosen_candidate: str = Field(..., min_length=1, max_length=200)
    suggested_by: str = Field("dashboard", max_length=100)


class ReviewLabelSubmission(BaseModel):
    id: int = Field(..., description="Row ID in labelling queue")
    label: str = Field(..., description="'1', '2', '3', 'none', or 'unsure'")
    labeller: str = Field(..., min_length=1, description="Student initials (e.g. 'AS', 'RK')")


# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> Any:
    """Serve the complete single-page interactive dashboard."""
    if not FRONTEND_FILE.exists():
        return HTMLResponse("<h2>Frontend building in progress...</h2>", status_code=200)
    return FileResponse(str(FRONTEND_FILE))


@app.get("/api/status", response_model=SystemStatus)
def get_status() -> SystemStatus:
    """Report pipeline status, active datasets, and labelling queue state."""
    conn = get_db_connection()
    try:
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        counts = {}
        for tbl in tables:
            cnt = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
            counts[tbl] = cnt

        hand_cnt = 0
        if HAND_LABELS_FILE.exists():
            df_h = pd.read_csv(HAND_LABELS_FILE)
            hand_cnt = len(df_h[df_h["label"].notna() & (df_h["label"] != "")])

        queue_cnt = 0
        if QUEUE_FILE.exists():
            df_q = pd.read_csv(QUEUE_FILE)
            queue_cnt = len(df_q)

        return SystemStatus(
            data_mode="REAL",
            database_connected=True,
            target_states=TARGET_STATES,
            banner="REAL DATA - data.gov.in - Official Open Government Data",
            table_counts=counts,
            labels_count=hand_cnt,
            queue_pending_count=max(0, queue_cnt - hand_cnt),
        )
    finally:
        conn.close()


@app.get("/api/catalog")
def get_catalog() -> List[Dict[str, Any]]:
    """Return dataset catalog with cryptographic lineage and portal URLs."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT table_name, row_count, source_url, resource_id, fetch_date, sha256 FROM meta_lineage"
        ).fetchall()
        return [
            {
                "table_name": r[0],
                "row_count": r[1],
                "source_url": r[2],
                "resource_id": r[3],
                "fetch_date": r[4],
                "sha256": r[5],
            }
            for r in rows
        ]
    finally:
        conn.close()


@app.get("/api/dq")
def get_dq(table: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return column-level data quality profiling metrics."""
    conn = get_db_connection()
    try:
        if table:
            rows = conn.execute(
                "SELECT table_name, column_name, semantic_type, total_rows, null_count, unique_count, notes "
                "FROM meta_data_quality WHERE table_name = ? ORDER BY column_name",
                [table],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT table_name, column_name, semantic_type, total_rows, null_count, unique_count, notes "
                "FROM meta_data_quality ORDER BY table_name, column_name"
            ).fetchall()

        return [
            {
                "table_name": r[0],
                "column_name": r[1],
                "semantic_type": r[2],
                "total_rows": r[3],
                "null_count": r[4],
                "unique_count": r[5],
                "notes": r[6] or "",
            }
            for r in rows
        ]
    finally:
        conn.close()


@app.get("/api/graph")
def get_dataset_graph() -> Dict[str, Any]:
    """Retrieve NetworkX integration graph nodes and validated join edges."""
    return build_dataset_graph_json(db_path=DB_PATH)


@app.get("/api/districts")
def get_integrated_districts(state: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return integrated district mart records with socioeconomic, economic, and postal metrics."""
    conn = get_db_connection()
    try:
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        if "integrated_districts" not in tables:
            return []

        if state:
            df = conn.execute(
                "SELECT * FROM integrated_districts WHERE UPPER(norm_state) = UPPER(?) ORDER BY norm_district",
                [state.strip()],
            ).fetchdf()
        else:
            df = conn.execute("SELECT * FROM integrated_districts ORDER BY norm_state, norm_district").fetchdf()

        return df.fillna(0).to_dict(orient="records")
    finally:
        conn.close()


@app.get("/api/ev/coverage")
def get_ev_coverage() -> Dict[str, Any]:
    """Provide state-grain comparative metrics across PCS 2024, OMC 2025, and Vahan 2023."""
    conn = get_db_connection()
    try:
        states_data = get_ev_state_comparison(conn)
        corridors = get_highway_corridors(conn)
        limitations = get_data_limitations_notice()
        return {
            "states": states_data,
            "corridors": corridors,
            "limitations": limitations,
        }
    finally:
        conn.close()


@app.get("/api/ev/reconciliation")
def get_ev_reconciliation() -> List[Dict[str, Any]]:
    """Audit arithmetic reconciliation between portal-reported totals and state row sums."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT table_name, reported_total, calculated_sum, difference, status FROM meta_ev_reconciliation"
        ).fetchall()
        return [
            {
                "table_name": r[0],
                "reported_total": r[1],
                "calculated_sum": r[2],
                "difference": r[3],
                "status": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()



@app.get("/api/review-queue")
def get_review_queue() -> List[Dict[str, Any]]:
    """Load items from labelling queue and merge any existing human annotations."""
    if not QUEUE_FILE.exists():
        return []

    df_queue = pd.read_csv(QUEUE_FILE, dtype={"label": str, "labeller": str})
    df_queue["label"] = df_queue["label"].fillna("")
    df_queue["labeller"] = df_queue["labeller"].fillna("")

    # Check if existing hand labels exist to overlay
    if HAND_LABELS_FILE.exists():
        df_labels = pd.read_csv(HAND_LABELS_FILE, dtype={"label": str, "labeller": str})
        df_labels["label"] = df_labels["label"].fillna("")
        df_labels["labeller"] = df_labels["labeller"].fillna("")
        label_map = df_labels.set_index("id")[["label", "labeller"]].to_dict(orient="index")
        for idx, row in df_queue.iterrows():
            row_id = row["id"]
            if row_id in label_map:
                df_queue.at[idx, "label"] = str(label_map[row_id].get("label", ""))
                df_queue.at[idx, "labeller"] = str(label_map[row_id].get("labeller", ""))

    return df_queue.astype(object).fillna("").to_dict(orient="records")


@app.post("/api/review/accept")
def submit_review_label(submission: ReviewLabelSubmission) -> Dict[str, Any]:
    """
    Record a student's hand-annotation for a district candidate.
    Appends or updates data/labels/hand_labels.csv adhering strictly to Section 9 Protocol.
    """
    valid_labels = {"1", "2", "3", "none", "unsure"}
    clean_lbl = str(submission.label).strip().lower()
    if clean_lbl not in valid_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label '{clean_lbl}'. Must be one of: 1, 2, 3, none, unsure.",
        )

    if not QUEUE_FILE.exists():
        raise HTTPException(status_code=404, detail="Labelling queue does not exist. Run resolver.py first.")

    df_queue = pd.read_csv(QUEUE_FILE, dtype={"label": str, "labeller": str})
    match_row = df_queue[df_queue["id"] == submission.id]
    if match_row.empty:
        raise HTTPException(status_code=404, detail=f"Item ID {submission.id} not found in labelling queue.")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if HAND_LABELS_FILE.exists():
        df_hand = pd.read_csv(HAND_LABELS_FILE, dtype={"label": str, "labeller": str})
    else:
        df_hand = df_queue.copy()

    df_hand["label"] = df_hand["label"].astype(object).fillna("")
    df_hand["labeller"] = df_hand["labeller"].astype(object).fillna("")

    # Update row in hand_labels
    row_idx = df_hand[df_hand["id"] == submission.id].index
    if not row_idx.empty:
        df_hand.loc[row_idx, "label"] = clean_lbl
        df_hand.loc[row_idx, "labeller"] = submission.labeller.strip().upper()
    else:
        new_row = match_row.iloc[0].to_dict()
        new_row["label"] = clean_lbl
        new_row["labeller"] = submission.labeller.strip().upper()
        df_hand = pd.concat([df_hand, pd.DataFrame([new_row])], ignore_index=True)

    df_hand.to_csv(HAND_LABELS_FILE, index=False)
    logger.info(f"Recorded hand label for item {submission.id} by {submission.labeller}")

    labelled_count = len(df_hand[df_hand["label"].astype(str).str.strip().isin(valid_labels)])

    return {
        "status": "success",
        "id": submission.id,
        "label": clean_lbl,
        "labeller": submission.labeller.strip().upper(),
        "total_labelled_count": labelled_count,
        "min_required_for_eval": 30,
    }


@app.post("/api/resolve", response_model=ResolveResponse)
def resolve_entity(req: ResolveRequest) -> ResolveResponse:
    """Resolve a district and expose candidate evidence; scores are similarities, not probabilities."""
    conn = get_db_connection()
    try:
        ref_dict = load_canonical_reference_from_db(conn)
        resolver = DistrictResolver(reference_dict=ref_dict)
        st, src = req.state.strip(), req.source_name.strip()
        if not src:
            raise HTTPException(status_code=422, detail="District name cannot be empty.")
        if st not in ref_dict:
            raise HTTPException(status_code=422, detail=f"No canonical reference data found for state: {st}")

        pred_ex, score_ex = resolver.baseline_exact(st, src)
        pred_tf, score_tf = resolver.baseline_tfidf(st, src)
        pred_rf, score_rf = resolver.baseline_rapidfuzz(st, src)
        candidates = resolver.get_top_candidates(st, src, k=5)
        cand_dicts = []
        for name, score in candidates:
            feats = compute_similarity_features(src, name)
            cand_dicts.append({
                "candidate": name, "score": round(float(score), 4),
                "evidence": {
                    "token_sort_similarity": round(feats["rf_token_sort"], 4),
                    "character_similarity": round(feats["rf_ratio"], 4),
                    "partial_similarity": round(feats["rf_partial"], 4),
                    "consonant_skeleton_similarity": round(feats["skeleton_sim"], 4),
                    "first_character_matches": bool(feats["first_char_match"]),
                    "length_difference": int(feats["len_diff"]),
                },
            })
        best_cand = candidates[0][0] if candidates else None
        best_score = float(candidates[0][1]) if candidates else 0.0
        if best_score >= req.threshold:
            band = "AUTO_LINK (Threshold Met)"
        elif best_score >= 0.50:
            band = "REVIEW_QUEUE (Uncertain Band)"
        else:
            band = "UNRESOLVED (Low Similarity)"
        return ResolveResponse(
            state=st, source_name=src,
            baseline_exact=pred_ex, baseline_exact_score=float(score_ex),
            baseline_tfidf=pred_tf, baseline_tfidf_score=float(score_tf),
            baseline_rapidfuzz=pred_rf, baseline_rapidfuzz_score=float(score_rf),
            top_candidates=cand_dicts, model_prediction=best_cand,
            model_confidence=best_score, decision_band=band,
        )
    finally:
        conn.close()


@app.post("/api/resolve/feedback")
def save_resolution_feedback(submission: CorrectionSubmission) -> Dict[str, Any]:
    """Save user corrections separately from evaluation labels to avoid contaminating ground truth."""
    conn = get_db_connection()
    try:
        refs = load_canonical_reference_from_db(conn)
        state = submission.state.strip()
        candidate = submission.chosen_candidate.strip()
        if state not in refs or candidate.lower() not in {x.lower() for x in refs[state]}:
            raise HTTPException(status_code=422, detail="Chosen candidate must exist in the canonical directory for the selected state.")
    finally:
        conn.close()
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": submission.state.strip(), "source_name": submission.source_name.strip(),
        "chosen_candidate": submission.chosen_candidate.strip(),
        "suggested_by": submission.suggested_by.strip() or "dashboard",
    }
    pd.DataFrame([row]).to_csv(FEEDBACK_FILE, mode="a", header=not FEEDBACK_FILE.exists(), index=False)
    return {"status": "saved", "message": "Correction saved for review; it has not been added to evaluation labels."}


@app.get("/api/analytics/quality")
def get_quality_analytics() -> Dict[str, Any]:
    """Aggregate existing DuckDB quality profiles and transformation logs for dashboard charts."""
    conn = get_db_connection()
    try:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "meta_data_quality" not in tables:
            return {"status": "unavailable", "datasets": [], "columns": [], "transformations": []}
        profiles = conn.execute("""
            SELECT table_name, column_name, semantic_type, total_rows, null_count, unique_count, notes
            FROM meta_data_quality ORDER BY table_name, column_name
        """).fetchdf()
        columns = []
        for row in profiles.to_dict(orient="records"):
            total = int(row.get("total_rows") or 0)
            nulls = int(row.get("null_count") or 0)
            columns.append({
                "table_name": str(row["table_name"]), "column_name": str(row["column_name"]),
                "semantic_type": str(row.get("semantic_type") or "unknown"),
                "total_rows": total, "null_count": nulls,
                "unique_count": int(row.get("unique_count") or 0),
                "missing_pct": round(100 * nulls / total, 2) if total else 0.0,
                "completeness_pct": round(100 * (total - nulls) / total, 2) if total else 100.0,
                "notes": str(row.get("notes") or ""),
            })
        grouped = {}
        for col in columns:
            item = grouped.setdefault(col["table_name"], {"table_name": col["table_name"], "columns": 0, "total_cells": 0, "missing_cells": 0, "row_count": 0})
            item["columns"] += 1
            item["total_cells"] += col["total_rows"]
            item["missing_cells"] += col["null_count"]
            item["row_count"] = max(item["row_count"], col["total_rows"])
        datasets = []
        for item in grouped.values():
            item["completeness_pct"] = round(100 * (item["total_cells"] - item["missing_cells"]) / item["total_cells"], 2) if item["total_cells"] else 100.0
            item["missing_pct"] = round(100 - item["completeness_pct"], 2)
            datasets.append(item)
        transformations = []
        if "meta_transform_log" in tables:
            transforms = conn.execute("SELECT step, table_name, description, rows_in, rows_out FROM meta_transform_log ORDER BY step").fetchall()
            transformations = [{"step": r[0], "table_name": r[1], "description": r[2], "rows_in": r[3], "rows_out": r[4], "rows_removed": max(0, int(r[3] or 0) - int(r[4] or 0))} for r in transforms]
        return {"status": "ok", "datasets": datasets, "columns": columns, "transformations": transformations}
    finally:
        conn.close()


@app.post("/api/upload/resolve")
async def resolve_uploaded_csv(
    file: UploadFile = File(...),
    state_column: str = Form(...),
    name_column: str = Form(...),
    threshold: float = Form(0.85),
) -> Dict[str, Any]:
    """Resolve uploaded CSV rows; returns JSON suitable for review and CSV export in the UI."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Threshold must be between 0 and 1.")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 10 MB upload limit.")
    try:
        df = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    if len(df) > 10000:
        raise HTTPException(status_code=413, detail="CSV is limited to 10,000 rows per run.")
    if state_column not in df.columns or name_column not in df.columns:
        raise HTTPException(status_code=422, detail=f"Choose columns present in the CSV. Available: {', '.join(map(str, df.columns))}")
    conn = get_db_connection()
    try:
        refs = load_canonical_reference_from_db(conn)
    finally:
        conn.close()
    resolver = DistrictResolver(reference_dict=refs)
    state_lookup = {str(key).casefold(): key for key in refs}
    results = []
    for index, row in df.iterrows():
        raw_state = str(row[state_column]).strip()
        state = state_lookup.get(raw_state.casefold(), raw_state)
        source_name = str(row[name_column]).strip()
        item = {"row_number": int(index) + 2, "state": raw_state, "source_name": source_name, "prediction": None, "similarity_score": 0.0, "decision": "UNRESOLVED", "top_candidates": []}
        if state in refs and source_name:
            candidates = resolver.get_top_candidates(state, source_name, k=3)
            if candidates:
                name, score = candidates[0]
                item.update({"prediction": name, "similarity_score": round(float(score), 4), "decision": "AUTO_LINK" if score >= threshold else ("REVIEW" if score >= 0.50 else "UNRESOLVED"), "top_candidates": [{"candidate": c, "score": round(float(sc), 4)} for c, sc in candidates]})
        elif state not in refs:
            item["decision"] = "UNKNOWN_STATE"
        results.append(item)
    return {"filename": file.filename, "rows_processed": len(results), "threshold": threshold, "results": results}


@app.get("/api/reports/errors")
def get_suspected_errors() -> List[Dict[str, str]]:
    """Return register of suspected anomalies found across data.gov.in tables."""
    return generate_suspected_error_report()


@app.get("/api/reports/release")
def download_release_archive() -> Any:
    """Generate and return complete release export ZIP file."""
    conn = get_db_connection()
    try:
        zip_file = generate_release_zip(conn)
        return FileResponse(
            str(zip_file),
            filename="south_india_engine_release.zip",
            media_type="application/zip",
        )
    finally:
        conn.close()


@app.get("/api/eval")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Return measured evaluation results only when current hand-label count meets the evaluation gate."""
    valid_count = 0
    if HAND_LABELS_FILE.exists():
        try:
            labels = pd.read_csv(HAND_LABELS_FILE, dtype=str)
            if "label" in labels.columns:
                valid_count = int(labels["label"].fillna("").str.strip().str.lower().isin(["1", "2", "3", "none"]).sum())
        except Exception:
            valid_count = 0
    if valid_count < 30:
        return {
            "status": "Awaiting Hand Labels",
            "message": f"Model comparison is gated until at least 30 valid hand labels are available. Current labelled rows: {valid_count}.",
            "labelled_count": valid_count, "min_required": 30,
            "remaining": 30 - valid_count,
        }
    if not EVAL_RESULTS_FILE.exists():
        return {"status": "Evaluation Not Run", "message": "Run `python evaluate.py` to calculate metrics from the current labelled dataset.", "labelled_count": valid_count, "min_required": 30}
    with open(EVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    current_fingerprint = hashlib.sha256(HAND_LABELS_FILE.read_bytes()).hexdigest() if HAND_LABELS_FILE.exists() else None
    if data.get("labels_sha256") and current_fingerprint != data.get("labels_sha256"):
        return {"status": "Evaluation Not Current", "message": "Hand labels have changed since these metrics were computed. Run `python evaluate.py` to refresh the comparison.", "labelled_count": valid_count, "min_required": 30}
    if int(data.get("sample_size_n", -1)) != valid_count:
        return {"status": "Evaluation Not Current", "message": "The evaluated sample size differs from the current hand-label count. Run `python evaluate.py` to refresh the comparison.", "labelled_count": valid_count, "min_required": 30}
    data["labelled_count"] = valid_count
    return data
