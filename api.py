"""
South India Open Data Integration Engine - REST API
FastAPI backend exposing data catalog, data quality profiling, resolver operations,
dataset integration graph, EV coverage explorer, interactive review queue, and reports.
Strictly parameterised DuckDB SELECT queries; no secrets in logs or responses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
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
    source_name: str = Field(..., description="Raw district string to match")


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
    """Test resolution of an arbitrary district string against canonical references."""
    conn = get_db_connection()
    try:
        ref_dict = load_canonical_reference_from_db(conn)
        resolver = DistrictResolver(reference_dict=ref_dict)

        st = req.state.strip()
        src = req.source_name.strip()

        # Baselines
        pred_ex, score_ex = resolver.baseline_exact(st, src)
        pred_tf, score_tf = resolver.baseline_tfidf(st, src)
        pred_rf, score_rf = resolver.baseline_rapidfuzz(st, src)

        # Candidates
        candidates = resolver.get_top_candidates(st, src, k=3)
        cand_dicts = [{"candidate": c[0], "score": c[1]} for c in candidates]

        best_cand = candidates[0][0] if candidates else None
        best_score = candidates[0][1] if candidates else 0.0

        if best_score >= 0.85:
            band = "AUTO_LINK (High Confidence)"
        elif best_score >= 0.50:
            band = "REVIEW_QUEUE (Uncertain Band)"
        else:
            band = "UNRESOLVED (Low Confidence)"

        return ResolveResponse(
            state=st,
            source_name=src,
            baseline_exact=pred_ex,
            baseline_exact_score=score_ex,
            baseline_tfidf=pred_tf,
            baseline_tfidf_score=score_tf,
            baseline_rapidfuzz=pred_rf,
            baseline_rapidfuzz_score=score_rf,
            top_candidates=cand_dicts,
            model_prediction=best_cand,
            model_confidence=best_score,
            decision_band=band,
        )
    finally:
        conn.close()


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
    """Retrieve evaluation metrics and confidence intervals from results.json."""
    if not EVAL_RESULTS_FILE.exists():
        return {
            "status": "Awaiting Hand Labels",
            "message": "Evaluation requires minimum 30 hand labels. Complete annotations in the Review Queue tab.",
            "min_required": 30,
        }
    with open(EVAL_RESULTS_FILE, "r") as f:
        return json.load(f)
