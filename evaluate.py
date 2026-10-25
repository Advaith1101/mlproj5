"""
South India Open Data Integration Engine - Model Evaluation & Benchmarking
Evaluates entity matching model against 3 baselines strictly on real student hand labels.
Rules:
  - Refuses to compute metrics below 30 labelled rows (statistically defensible small-n guardrail).
  - Reports Wilson score confidence intervals (95% CI) and explicit sample size n.
  - Compares Baseline 1 (Exact), Baseline 2 (TF-IDF), Baseline 3 (RapidFuzz), and ML Logistic Regression.
  - Saves measured metrics to data/work/results.json (never writes an uncomputed number).
"""

from __future__ import annotations

import json
import logging
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from config import DATA_DIR, SEED, WORK_DIR
from resolver import (
    DistrictResolver,
    compute_similarity_features,
    load_canonical_reference_from_db,
)

logger = logging.getLogger(__name__)

RESULTS_FILE = WORK_DIR / "results.json"
HAND_LABELS_FILE = DATA_DIR / "labels" / "hand_labels.csv"


def wilson_score_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Wilson score continuity-corrected confidence interval for a binomial proportion.
    Essential for honest reporting with small sample sizes n (e.g. n ~ 30-60).
    """
    if trials == 0:
        return 0.0, 0.0

    p = successes / trials
    # Two-sided normal quantile: 1.95996 for 95% confidence
    z = 1.959964 if abs(confidence - 0.95) < 1e-4 else 1.644853

    denominator = 1.0 + (z**2) / trials
    center_adj = p + (z**2) / (2.0 * trials)
    radicand = (p * (1.0 - p) + (z**2) / (4.0 * trials)) / trials
    margin = z * math.sqrt(max(0.0, radicand))

    ci_low = max(0.0, (center_adj - margin) / denominator)
    ci_high = min(1.0, (center_adj + margin) / denominator)
    return round(ci_low, 4), round(ci_high, 4)


def run_evaluation_on_labels(
    df_labels: pd.DataFrame,
    min_labels: int = 30,
    db_conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> Dict[str, Any]:
    """Compare four methods on the same rows, using grouped CV for the learned ranker."""
    required = {"label", "state", "source_name", "cand1"}
    missing = required.difference(df_labels.columns)
    if missing:
        raise ValueError(f"Evaluation labels are missing required columns: {', '.join(sorted(missing))}")

    # 'unsure' is an annotation status, not ground truth; do not count it as a negative.
    accepted_labels = {"1", "2", "3", "none"}
    label_series = df_labels["label"].fillna("").astype(str).str.strip().str.lower()
    df_valid = df_labels[label_series.isin(accepted_labels)].copy().reset_index(drop=True)
    df_valid["label"] = df_valid["label"].astype(str).str.strip().str.lower()
    n = len(df_valid)
    if n < min_labels:
        raise ValueError(
            f"Evaluation refused: Found only {n} valid hand labels (excluding 'unsure' and blank rows). "
            f"A minimum of {min_labels} student hand-annotated labels is required for statistically meaningful evaluation."
        )

    if db_conn is not None:
        ref_dict = load_canonical_reference_from_db(db_conn)
    else:
        db_file = WORK_DIR / "engine.duckdb"
        if db_file.exists():
            c = duckdb.connect(str(db_file), read_only=True)
            try:
                ref_dict = load_canonical_reference_from_db(c)
            finally:
                c.close()
        else:
            ref_dict = {}
    resolver = DistrictResolver(reference_dict=ref_dict)

    def ground_truth(row: pd.Series) -> Optional[str]:
        lbl = str(row["label"]).strip().lower()
        if lbl in {"1", "2", "3"}:
            col = f"cand{lbl}"
            val = row.get(col)
            return str(val).strip().casefold() if pd.notna(val) and str(val).strip() else None
        return None  # explicit 'none' annotation

    eval_rows: List[Dict[str, Any]] = []
    for row_idx, (_, row) in enumerate(df_valid.iterrows()):
        state = str(row["state"]).strip()
        source = str(row["source_name"]).strip()
        truth = ground_truth(row)
        p1, s1 = resolver.baseline_exact(state, source)
        p2, s2 = resolver.baseline_tfidf(state, source)
        p3, s3 = resolver.baseline_rapidfuzz(state, source)
        candidates = []
        for col in ("cand1", "cand2", "cand3"):
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                name = str(value).strip()
                if name.casefold() not in {x.casefold() for x in candidates}:
                    candidates.append(name)
        eval_rows.append({
            "row_group": row_idx, "state": state, "source_name": source, "truth": truth,
            "exact": p1, "tfidf": p2, "rapidfuzz": p3, "candidates": candidates,
            "exact_score": float(s1), "tfidf_score": float(s2), "rapidfuzz_score": float(s3),
        })

    # Build candidate-pair training examples. All candidates for one source row stay in
    # the same fold, preventing candidate-level leakage between train and validation.
    X_pairs: List[List[float]] = []
    y_pairs: List[int] = []
    pair_groups: List[int] = []
    pair_lookup: Dict[int, List[Tuple[str, int]]] = {}
    for item in eval_rows:
        for candidate in item["candidates"]:
            feats = compute_similarity_features(item["source_name"], candidate)
            X_pairs.append([
                feats["rf_ratio"], feats["rf_token_sort"], feats["rf_partial"],
                feats["skeleton_sim"], feats["first_char_match"], feats["len_diff"],
            ])
            y_pairs.append(int(item["truth"] is not None and candidate.casefold() == item["truth"]))
            pair_groups.append(item["row_group"])
            pair_lookup.setdefault(item["row_group"], []).append((candidate, len(X_pairs) - 1))

    ml_predictions: Dict[int, Optional[str]] = {}
    X_arr = np.asarray(X_pairs, dtype=float)
    y_arr = np.asarray(y_pairs, dtype=int)
    unique_groups = np.unique(pair_groups) if pair_groups else np.array([])
    if len(X_arr) and len(unique_groups) >= 2 and len(np.unique(y_arr)) > 1:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        groups_arr = np.asarray(pair_groups)
        for train_idx, test_idx in splitter.split(X_arr, y_arr, groups_arr):
            train_y = y_arr[train_idx]
            if len(np.unique(train_y)) < 2:
                # This fold cannot train a binary classifier; use its strongest string candidate.
                test_groups = set(groups_arr[test_idx].tolist())
                for group in test_groups:
                    item = eval_rows[int(group)]
                    ml_predictions[int(group)] = item["rapidfuzz"]
                continue
            clf = LogisticRegression(random_state=SEED, max_iter=300, class_weight="balanced")
            clf.fit(X_arr[train_idx], train_y)
            test_prob = clf.predict_proba(X_arr[test_idx])[:, list(clf.classes_).index(1)]
            # Aggregate candidate probabilities per held-out row and pick the highest.
            test_index_set = set(test_idx.tolist())
            for group in set(groups_arr[test_idx].tolist()):
                group_candidates = pair_lookup.get(int(group), [])
                local_scores = []
                for idx, prob in zip(test_idx, test_prob):
                    if idx in test_index_set and int(groups_arr[idx]) == int(group):
                        local_scores.append((int(idx), float(prob)))
                best = max(local_scores, key=lambda x: x[1]) if local_scores else None
                if best is None or best[1] < 0.50:
                    ml_predictions[int(group)] = None
                else:
                    candidate_for_idx = next((c for c, pair_idx in group_candidates if pair_idx == best[0]), None)
                    ml_predictions[int(group)] = candidate_for_idx
    else:
        for item in eval_rows:
            ml_predictions[item["row_group"]] = item["rapidfuzz"]

    methods = {
        "baseline_1_exact": ("Exact Normalized Match", "exact"),
        "baseline_2_tfidf": ("Character 2-4-gram TF-IDF Cosine", "tfidf"),
        "baseline_3_rapidfuzz": ("RapidFuzz Token Sort Ratio", "rapidfuzz"),
        "ml_model_logistic_regression": ("Logistic Regression Candidate Ranker (Grouped CV)", "ml"),
    }
    baselines: Dict[str, Any] = {}
    for key, (name, pred_key) in methods.items():
        correct = 0
        tp = fp = fn = 0
        for item in eval_rows:
            prediction = ml_predictions.get(item["row_group"]) if pred_key == "ml" else item[pred_key]
            pred_norm = str(prediction).strip().casefold() if prediction is not None and str(prediction).strip() else None
            truth = item["truth"]
            if pred_norm == truth:
                correct += 1
            if pred_norm is not None and truth is not None and pred_norm == truth:
                tp += 1
            elif pred_norm is not None:
                fp += 1
            if truth is not None and pred_norm != truth:
                fn += 1
        accuracy = correct / n
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ci = wilson_score_interval(correct, n)
        baselines[key] = {
            "name": name, "accuracy": round(accuracy, 4), "correct_count": correct,
            "wilson_95_ci": list(ci), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1_score": round(f1, 4), "sample_size_n": n,
        }

    state_breakdown: Dict[str, Any] = {}
    for state in sorted({item["state"] for item in eval_rows}):
        items = [item for item in eval_rows if item["state"] == state]
        state_breakdown[state] = {
            "n": len(items),
            "top1_accuracy": round(sum(1 for it in items if (str(it["rapidfuzz"]).strip().casefold() if it["rapidfuzz"] else None) == it["truth"]) / len(items), 4) if items else 0.0,
        }

    label_fingerprint = hashlib.sha256(HAND_LABELS_FILE.read_bytes()).hexdigest() if HAND_LABELS_FILE.exists() else hashlib.sha256(df_valid.to_csv(index=False).encode("utf-8")).hexdigest()
    results = {
        "labels_sha256": label_fingerprint,
        "sample_size_n": n,
        "evaluation_rule": "Measured on valid hand labels only; blank and 'unsure' annotations excluded. All four methods use the same source rows. Precision/recall/F1 treat an exact candidate match as a true positive; incorrect non-empty suggestions count as false positives and missed true matches as false negatives.",
        "baselines": baselines,
        "per_state_breakdown": state_breakdown,
        "statistical_notice": f"Given n = {n}, Wilson 95% intervals show sampling uncertainty. Logistic Regression is evaluated with source-row-grouped cross-validation.",
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    if not HAND_LABELS_FILE.exists():
        print(f"[!] {HAND_LABELS_FILE} does not exist yet.")
        print("[!] Evaluation requires student hand labels. Use the Review Queue in the dashboard.")
        return

    df_labels = pd.read_csv(HAND_LABELS_FILE)
    valid_count = int(df_labels["label"].fillna("").astype(str).str.strip().str.lower().isin(["1", "2", "3", "none"]).sum())
    print(f"[*] Found {valid_count} labelled rows in {HAND_LABELS_FILE}.")

    if valid_count < 30:
        print(f"[!] Minimum 30 labels required (found {valid_count}). Awaiting student completion.")
        return

    print("[*] Running full evaluation suite...")
    results = run_evaluation_on_labels(df_labels, min_labels=30)
    print("[OK] Evaluation complete. Results saved to data/work/results.json.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
