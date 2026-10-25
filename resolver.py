"""
South India Open Data Integration Engine - District Entity Matching & Resolver
Resolves inconsistent district names across ministries (MGNREGA, MSME) to
the canonical Department of Posts Pincode directory reference list.
Enforces strict state-level blocking (never cross state borders).
Includes:
  - Orthographic & phonetic feature extraction (consonant skeleton, token sort)
  - 3 Baselines: (1) Exact Normalized, (2) Char n-gram TF-IDF, (3) RapidFuzz
  - Machine Learning Classifier: Scikit-Learn Logistic Regression with calibrated probability
  - Labelling Queue Generator (Section 9 Protocol)
  - Suspected District Reorganisation/Split Detector
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

from config import DATA_DIR, SEED, WORK_DIR

logger = logging.getLogger(__name__)

VOWELS = set("aeiou")


def consonant_skeleton(text: Optional[str]) -> str:
    """
    Extract compressed consonant skeleton preserving phonetic core.
    Removes vowels, punctuation, and collapses consecutive identical consonants.
    Example: 'bengaluru' -> 'bnglr', 'kasaragod' -> 'ksrgd', 'chennai' -> 'chn'.
    """
    if not text:
        return ""
    s = unidecode(str(text)).lower().strip()
    # Retain only alphabetic letters
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""
    # Remove vowels
    consonants = [ch for ch in s if ch not in VOWELS]
    # Collapse consecutive duplicates (e.g. 'nn' -> 'n')
    collapsed = []
    for ch in consonants:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return "".join(collapsed)


def compute_similarity_features(
    s1: str, s2: str, tfidf_sim: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute pairwise orthographic, phonetic, and length features between two district names.
    Returns feature dictionary used by the ML resolver.
    """
    str1 = str(s1).lower().strip()
    str2 = str(s2).lower().strip()

    # RapidFuzz string metrics
    rf_ratio = fuzz.ratio(str1, str2) / 100.0
    rf_token_sort = fuzz.token_sort_ratio(str1, str2) / 100.0
    rf_partial = fuzz.partial_ratio(str1, str2) / 100.0

    # Consonant skeleton similarity
    skel1 = consonant_skeleton(str1)
    skel2 = consonant_skeleton(str2)
    skel_sim = (fuzz.ratio(skel1, skel2) / 100.0) if (skel1 and skel2) else 0.0

    # First letter agreement
    first_char_match = 1.0 if (str1 and str2 and str1[0] == str2[0]) else 0.0

    # Absolute length difference
    len_diff = float(abs(len(str1) - len(str2)))

    features = {
        "rf_ratio": rf_ratio,
        "rf_token_sort": rf_token_sort,
        "rf_partial": rf_partial,
        "skeleton_sim": skel_sim,
        "first_char_match": first_char_match,
        "len_diff": len_diff,
    }
    if tfidf_sim is not None:
        features["tfidf_sim"] = tfidf_sim
    return features


class DistrictResolver:
    """
    Entity matching resolver with state blocking and tiered decision bands:
      - Auto-link: high confidence (probability >= 0.85)
      - Review Queue: uncertain band (0.50 <= probability < 0.85)
      - Unresolved: low confidence (probability < 0.50)
    """

    def __init__(self, reference_dict: Optional[Dict[str, List[str]]] = None) -> None:
        """
        reference_dict: mapping from State Name -> List of canonical district names.
        """
        self.reference_dict: Dict[str, List[str]] = reference_dict or {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.ref_matrices: Dict[str, Any] = {}
        self.model: Optional[LogisticRegression] = None
        self._init_vectorizers()

    def _init_vectorizers(self) -> None:
        """Fit character n-gram (2-4) TF-IDF vectorizers per state for fast candidate retrieval."""
        for state, districts in self.reference_dict.items():
            if districts:
                vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
                mat = vec.fit_transform(districts)
                self.vectorizers[state] = vec
                self.ref_matrices[state] = mat

    def set_reference_dict(self, ref_dict: Dict[str, List[str]]) -> None:
        self.reference_dict = ref_dict
        self._init_vectorizers()

    # -----------------------------------------------------------------
    # BASELINES
    # -----------------------------------------------------------------
    def baseline_exact(self, state: str, source_name: str) -> Tuple[Optional[str], float]:
        """Baseline 1: Exact normalized string match within state blocking."""
        s = str(source_name).lower().strip()
        ref_list = self.reference_dict.get(state, [])
        for ref in ref_list:
            if s == ref.lower().strip():
                return ref, 1.0
        return None, 0.0

    def baseline_tfidf(self, state: str, source_name: str) -> Tuple[Optional[str], float]:
        """Baseline 2: Character 2-4-gram TF-IDF cosine similarity."""
        s = str(source_name).lower().strip()
        if state not in self.vectorizers or not self.reference_dict.get(state):
            return None, 0.0
        vec = self.vectorizers[state]
        mat = self.ref_matrices[state]
        src_vec = vec.transform([s])
        sims = cosine_similarity(src_vec, mat).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return self.reference_dict[state][best_idx], best_score

    def baseline_rapidfuzz(self, state: str, source_name: str) -> Tuple[Optional[str], float]:
        """Baseline 3: RapidFuzz token sort ratio."""
        s = str(source_name).lower().strip()
        ref_list = self.reference_dict.get(state, [])
        if not ref_list:
            return None, 0.0
        best_cand = None
        best_score = -1.0
        for ref in ref_list:
            score = fuzz.token_sort_ratio(s, ref.lower().strip()) / 100.0
            if score > best_score:
                best_score = score
                best_cand = ref
        return best_cand, best_score

    # -----------------------------------------------------------------
    # CANDIDATE GENERATION (Strict State Blocking)
    # -----------------------------------------------------------------
    def get_top_candidates(
        self, state: str, source_name: str, k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k reference candidate districts strictly within the same state.
        Ranked using RapidFuzz composite similarity.
        """
        ref_list = self.reference_dict.get(state, [])
        if not ref_list:
            return []

        s = str(source_name).lower().strip()
        candidates = []
        for ref in ref_list:
            r = ref.lower().strip()
            # Score combining token sort and skeleton similarity
            ts_score = fuzz.token_sort_ratio(s, r) / 100.0
            sk_score = (
                fuzz.ratio(consonant_skeleton(s), consonant_skeleton(r)) / 100.0
            )
            composite = round(0.7 * ts_score + 0.3 * sk_score, 4)
            candidates.append((ref, composite))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a lightweight, defensible Logistic Regression classifier."""
        clf = LogisticRegression(random_state=SEED, max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        self.model = clf

    def predict_pair_confidence(self, s1: str, s2: str) -> float:
        """Predict probability that two names represent the exact same district."""
        feats = compute_similarity_features(s1, s2)
        X = np.array([
            feats["rf_ratio"],
            feats["rf_token_sort"],
            feats["rf_partial"],
            feats["skeleton_sim"],
            feats["first_char_match"],
            feats["len_diff"],
        ]).reshape(1, -1)

        if self.model is not None:
            return float(self.model.predict_proba(X)[0][1])
        # Fallback heuristic if ML model is not yet trained
        return float(feats["rf_token_sort"])


def load_canonical_reference_from_db(conn: Any) -> Dict[str, List[str]]:
    """
    Extract canonical reference list of (norm_state, norm_district) from clean_pincode.
    """
    df = conn.execute(
        "SELECT DISTINCT norm_state, norm_district FROM clean_pincode "
        "WHERE norm_state IS NOT NULL AND norm_district IS NOT NULL "
        "ORDER BY norm_state, norm_district"
    ).fetchdf()

    ref_dict: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        st = row["norm_state"]
        dist = row["norm_district"]
        if st not in ref_dict:
            ref_dict[st] = []
        ref_dict[st].append(dist)
    return ref_dict


def build_labelling_queue_df(unresolved_items: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Format unresolved candidate matches strictly to Section 9 Protocol.
    Columns: id, source_table, state, source_name, cand1, cand2, cand3, score1, score2, score3, label, labeller.
    """
    rows = []
    for item in unresolved_items:
        cands = item.get("candidates", [])
        cand1 = cands[0][0] if len(cands) > 0 else ""
        score1 = round(cands[0][1], 4) if len(cands) > 0 else 0.0
        cand2 = cands[1][0] if len(cands) > 1 else ""
        score2 = round(cands[1][1], 4) if len(cands) > 1 else 0.0
        cand3 = cands[2][0] if len(cands) > 2 else ""
        score3 = round(cands[2][1], 4) if len(cands) > 2 else 0.0

        rows.append({
            "id": item["id"],
            "source_table": item["source_table"],
            "state": item["state"],
            "source_name": item["source_name"],
            "cand1": cand1,
            "cand2": cand2,
            "cand3": cand3,
            "score1": score1,
            "score2": score2,
            "score3": score3,
            "label": "",       # Left blank for students to label (1, 2, 3, none, unsure)
            "labeller": "",    # Left blank for student initials
        })

    return pd.DataFrame(rows)


def detect_suspected_renames_and_splits(conn: Any) -> pd.DataFrame:
    """
    Identify potential district bifurcations, renames, and mergers across datasets.
    Example: Andhra Pradesh MSME (13 districts) vs Pincode (26 districts bifurcated in 2022).
    """
    records = []

    # 1. Compare MSME vs Pincode district counts per state
    df_msme_counts = conn.execute(
        "SELECT norm_state, count(DISTINCT norm_district) as cnt FROM clean_msme GROUP BY norm_state"
    ).fetchdf().set_index("norm_state")["cnt"].to_dict()

    df_pin_counts = conn.execute(
        "SELECT norm_state, count(DISTINCT norm_district) as cnt FROM clean_pincode GROUP BY norm_state"
    ).fetchdf().set_index("norm_state")["cnt"].to_dict()

    for st in df_msme_counts:
        m_cnt = df_msme_counts[st]
        p_cnt = df_pin_counts.get(st, 0)
        if m_cnt != p_cnt:
            records.append({
                "state": st,
                "event_type": "District Count Divergence",
                "evidence": f"MSME table has {m_cnt} districts while Pincode Directory has {p_cnt} districts",
                "impact": "Suspected district reorganisation or historic panel alignment divergence.",
            })

    # 2. MGNREGA Financial Year appearance/disappearance
    df_mg_years = conn.execute(
        "SELECT DISTINCT norm_state, norm_district, fin_year FROM clean_mgnrega"
    ).fetchdf()

    years = sorted(df_mg_years["fin_year"].unique())
    if len(years) >= 2:
        earliest_yr = years[0]
        latest_yr = years[-1]
        early_dists = set(
            df_mg_years[df_mg_years["fin_year"] == earliest_yr]["norm_district"].unique()
        )
        late_dists = set(
            df_mg_years[df_mg_years["fin_year"] == latest_yr]["norm_district"].unique()
        )

        new_in_late = late_dists - early_dists
        missing_in_late = early_dists - late_dists

        if new_in_late:
            records.append({
                "state": "Multi-State (MGNREGA)",
                "event_type": "New Districts in Recent Financial Years",
                "evidence": f"{len(new_in_late)} district names present in {latest_yr} were absent in {earliest_yr}: {sorted(list(new_in_late))[:5]}...",
                "impact": "Indicates state district bifurcations during the reporting window.",
            })

    return pd.DataFrame(records)


def export_labelling_queue_from_db(
    conn: Any,
    labels_dir: Optional[Path] = None,
) -> Path:
    """
    Generate and export data/labels/labelling_queue.csv containing all distinct
    district names from MGNREGA and MSME that did not achieve an exact normalized match
    against the Department of Posts reference list.
    """
    out_dir = Path(labels_dir) if labels_dir else (DATA_DIR / "labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "labelling_queue.csv"

    ref_dict = load_canonical_reference_from_db(conn)
    resolver = DistrictResolver(reference_dict=ref_dict)

    unresolved_items = []
    item_id = 1

    # Scan MGNREGA distinct districts
    df_mg = conn.execute(
        "SELECT DISTINCT norm_state, norm_district FROM clean_mgnrega "
        "WHERE norm_state IS NOT NULL AND norm_district IS NOT NULL "
        "ORDER BY norm_state, norm_district"
    ).fetchdf()

    for _, row in df_mg.iterrows():
        st = row["norm_state"]
        dist = row["norm_district"]
        pred, score = resolver.baseline_exact(st, dist)
        if pred is None:
            cands = resolver.get_top_candidates(st, dist, k=3)
            unresolved_items.append({
                "id": item_id,
                "source_table": "clean_mgnrega",
                "state": st,
                "source_name": dist,
                "candidates": cands,
            })
            item_id += 1

    # Scan MSME distinct districts
    df_msme = conn.execute(
        "SELECT DISTINCT norm_state, norm_district FROM clean_msme "
        "WHERE norm_state IS NOT NULL AND norm_district IS NOT NULL "
        "ORDER BY norm_state, norm_district"
    ).fetchdf()

    for _, row in df_msme.iterrows():
        st = row["norm_state"]
        dist = row["norm_district"]
        pred, score = resolver.baseline_exact(st, dist)
        if pred is None:
            # Check if this state/dist pair was already captured from mgnrega
            existing = [u for u in unresolved_items if u["state"] == st and u["source_name"] == dist]
            if not existing:
                cands = resolver.get_top_candidates(st, dist, k=3)
                unresolved_items.append({
                    "id": item_id,
                    "source_table": "clean_msme",
                    "state": st,
                    "source_name": dist,
                    "candidates": cands,
                })
                item_id += 1

    df_queue = build_labelling_queue_df(unresolved_items)
    df_queue.to_csv(out_file, index=False)
    logger.info(f"Exported labelling queue with {len(df_queue)} rows to {out_file}")
    return out_file
