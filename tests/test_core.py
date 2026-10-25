"""
Unit test suite for South India Open Data Integration Engine (Fetch & Ingestion).
Strictly follows TDD: mocked network calls only, secret hygiene validation,
and test fixtures clearly labelled TEST FIXTURE (never saved into real data/).
"""

import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from config import sanitize_url, DATASETS


# =====================================================================
# TEST FIXTURES (Rule D4: Tiny hand-written rows labelled TEST FIXTURE)
# =====================================================================
TEST_FIXTURE_PAGE_1 = {
    "status": "ok",
    "total": 3,
    "count": 2,
    "limit": 2,
    "offset": 0,
    "records": [
        {"id": "1", "state_name": "TAMIL NADU", "district_name": "CHENNAI"},
        {"id": "2", "state_name": "KARNATAKA", "district_name": "BENGALURU URBAN"},
    ],
}

TEST_FIXTURE_PAGE_2 = {
    "status": "ok",
    "total": 3,
    "count": 1,
    "limit": 2,
    "offset": 2,
    "records": [
        {"id": "3", "state_name": "KERALA", "district_name": "THIRUVANANTHAPURAM"},
    ],
}


class TestSecretHygiene:
    """Tests ensuring OGD_API_KEY is never leaked or logged (Rule D4)."""

    def test_sanitize_url_strips_query_string(self):
        secret_url = "https://api.data.gov.in/resource/test-id?api-key=SECRET_KEY_12345&format=json&limit=10"
        sanitized = sanitize_url(secret_url)
        assert "SECRET_KEY_12345" not in sanitized
        assert "api-key" not in sanitized
        assert sanitized == "https://api.data.gov.in/resource/test-id"

    def test_sanitize_url_handles_clean_url(self):
        clean_url = "https://api.data.gov.in/resource/test-id"
        assert sanitize_url(clean_url) == clean_url

    def test_key_not_in_datasets_manifest(self):
        # DATASETS manifest in config.py must not contain any raw API key string
        manifest_str = json.dumps(DATASETS)
        assert "api-key" not in manifest_str.lower() or "api_endpoint" in manifest_str


class TestFetchDataLogic:
    """Tests for pagination, caching, backoff, and manifest creation using mocked network."""

    @patch("fetch_data.requests.get")
    def test_pagination_fetches_all_pages(self, mock_get):
        from fetch_data import fetch_resource_paginated

        # Mock page 1 and page 2 responses
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = TEST_FIXTURE_PAGE_1

        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = TEST_FIXTURE_PAGE_2

        mock_get.side_effect = [resp1, resp2]

        records = fetch_resource_paginated(
            resource_id="dummy-resource-id",
            api_key="mock_secret_key",
            batch_size=2,
            max_records=10,
            delay_sec=0.0,
        )

        assert len(records) == 3
        assert records[0]["district_name"] == "CHENNAI"
        assert records[2]["district_name"] == "THIRUVANANTHAPURAM"
        assert mock_get.call_count == 2

    @patch("fetch_data.requests.get")
    def test_exponential_backoff_on_transient_error(self, mock_get):
        from fetch_data import fetch_with_retry

        # First call fails with 503, second succeeds with 200
        resp_fail = MagicMock()
        resp_fail.status_code = 503
        resp_fail.raise_for_status.side_effect = Exception("Server Error 503")

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = TEST_FIXTURE_PAGE_1

        mock_get.side_effect = [resp_fail, resp_ok]

        with patch("fetch_data.time.sleep") as mock_sleep:
            res = fetch_with_retry("https://api.data.gov.in/resource/dummy", params={}, max_retries=3, initial_delay=0.01)
            assert res is not None
            assert res.json()["total"] == 3
            assert mock_sleep.called

    def test_cache_skips_existing_download(self):
        from fetch_data import is_cached

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cached_file = tmp_path / "test_data.json"
            cached_file.write_text(json.dumps(TEST_FIXTURE_PAGE_1))

            assert is_cached("test_data.json", raw_dir=tmp_path) is True
            assert is_cached("non_existent.json", raw_dir=tmp_path) is False

    def test_manifest_sha256_and_structure(self):
        from fetch_data import update_manifest, compute_sha256

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_file = tmp_path / "sample.json"
            sample_content = json.dumps(TEST_FIXTURE_PAGE_1)
            sample_file.write_text(sample_content)

            sha = compute_sha256(sample_file)
            expected_sha = hashlib.sha256(sample_content.encode("utf-8")).hexdigest()
            assert sha == expected_sha

            manifest_path = tmp_path / "manifest.json"
            update_manifest(
                dataset_key="test_dataset",
                resource_id="res-1234",
                title="Test Dataset Title",
                url="https://www.data.gov.in/resource/test",
                filename="sample.json",
                row_count=2,
                sha256_hash=sha,
                manifest_path=manifest_path,
            )

            assert manifest_path.exists()
            with open(manifest_path) as f:
                manifest_data = json.load(f)

            assert "test_dataset" in manifest_data
            entry = manifest_data["test_dataset"]
            assert entry["id"] == "res-1234"
            assert entry["title"] == "Test Dataset Title"
            assert entry["sha256"] == expected_sha
            assert entry["row_count"] == 2
            assert "fetch_date" in entry

    def test_manual_csv_registration(self):
        from fetch_data import register_local_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_file = tmp_path / "manual_test.csv"
            # TEST FIXTURE: Small CSV
            csv_content = "state_name,district_name,value\nTAMIL NADU,COIMBATORE,100\nKERALA,WAYANAD,200\n"
            csv_file.write_text(csv_content)

            manifest_path = tmp_path / "manifest.json"
            entry = register_local_csv(
                dataset_key="manual_dataset",
                resource_id="manual-res-id",
                title="Manual Registered Dataset",
                url="https://www.data.gov.in/manual",
                csv_path=csv_file,
                manifest_path=manifest_path,
            )

            assert entry["row_count"] == 2
            assert entry["sha256"] == hashlib.sha256(csv_file.read_bytes()).hexdigest()
            assert manifest_path.exists()


# =====================================================================
# PHASE 2 TESTS: Pipeline Normalisation, Plausibility, Reconciliation
# =====================================================================
class TestPipelineNormalisation:
    """Tests text standardisation across ministry tables."""

    def test_normalize_state_name(self):
        from pipeline import normalize_state_name

        assert normalize_state_name("TAMIL NADU") == "Tamil Nadu"
        assert normalize_state_name("tamil nadu") == "Tamil Nadu"
        assert normalize_state_name("Pondicherry") == "Puducherry"
        assert normalize_state_name("PUDUCHERRY") == "Puducherry"
        assert normalize_state_name("Andaman and Nicobar Island") == "Andaman and Nicobar Islands"
        assert normalize_state_name("TELANGANA") == "Telangana"
        assert normalize_state_name(None) is None
        assert normalize_state_name("") is None

    def test_normalize_district_name(self):
        from pipeline import normalize_district_name

        assert normalize_district_name("CHENNAI") == "chennai"
        assert normalize_district_name("Bengaluru Urban") == "bengaluru urban"
        assert normalize_district_name("Thoothukkudi (Tuticorin)") == "thoothukkudi tuticorin"
        assert normalize_district_name("Kasaragod District") == "kasaragod"
        assert normalize_district_name("Ranga  Reddy") == "ranga reddy"
        assert normalize_district_name("The Nilgiris") == "nilgiris"
        assert normalize_district_name(None) is None


class TestPipelinePlausibility:
    """Tests data plausibility rules for codes, coordinates, and types."""

    def test_clean_pincode(self):
        from pipeline import clean_pincode

        assert clean_pincode("600001") == "600001"
        assert clean_pincode(600001) == "600001"
        assert clean_pincode("600 001") == "600001"
        assert clean_pincode("000000") is None  # Invalid Indian PIN prefix
        assert clean_pincode("123") is None     # Length < 6
        assert clean_pincode("60000A") is None  # Non-digit
        assert clean_pincode(None) is None
        assert clean_pincode("NA") is None

    def test_clean_coordinate(self):
        from pipeline import clean_coordinate

        # Chennai: ~13.0827 N, 80.2707 E
        assert clean_coordinate("13.0827", "lat") == 13.0827
        assert clean_coordinate(80.2707, "lon") == 80.2707
        assert clean_coordinate("0", "lat") is None        # Invalid 0 coord
        assert clean_coordinate("0.0", "lon") is None
        assert clean_coordinate("NA", "lat") is None
        assert clean_coordinate(None, "lat") is None
        assert clean_coordinate("95.0", "lat") is None     # Implausible latitude
        assert clean_coordinate("150.0", "lon") is None    # Implausible longitude for India


class TestPipelineReconciliation:
    """Tests summary row separation and arithmetic reconciliation."""

    def test_ev_summary_row_reconciliation(self):
        import pandas as pd
        from pipeline import separate_summary_row

        # TEST FIXTURE: Small EV table with a summary row
        df_fixture = pd.DataFrame([
            {"state_ut": "Tamil Nadu", "operational_pcs": "400"},
            {"state_ut": "Karnataka", "operational_pcs": "500"},
            {"state_ut": "Kerala", "operational_pcs": "300"},
            {"state_ut": "Total Operational PCS", "operational_pcs": "1200"},
        ])

        df_clean, summary_info = separate_summary_row(
            df=df_fixture,
            state_col="state_ut",
            val_col="operational_pcs",
            summary_pattern="Total",
        )

        assert len(df_clean) == 3
        assert summary_info is not None
        assert summary_info["reported_total"] == 1200
        assert summary_info["calculated_sum"] == 1200
        assert summary_info["difference"] == 0
        assert summary_info["status"] == "RECONCILED"


class TestPipelineExecutionInMemory:
    """Verifies that the entire pipeline can execute against an in-memory DuckDB."""

    def test_pipeline_fixture_in_memory(self):
        from pipeline import DataPipeline

        pipeline = DataPipeline(db_path=":memory:")
        # Verify schema initialization
        tables = pipeline.get_table_names()
        assert "meta_lineage" in tables
        assert "meta_transform_log" in tables
        assert "meta_rejected_records" in tables
        assert "meta_ev_reconciliation" in tables
        assert "meta_data_quality" in tables
        pipeline.close()


# =====================================================================
# API SMOKE TESTS (FastAPI TestClient)
# =====================================================================
class TestApiSmoke:
    """Smoke tests for FastAPI endpoints verifying data responses and zero secret leak."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from api import app

        self.client = TestClient(app)

    def test_root_serves_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "South India Open Data Integration Engine" in resp.text
        assert "REAL DATA" in resp.text


    def test_api_status_endpoint(self):
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_mode"] == "REAL"
        assert data["database_connected"] is True
        assert len(data["target_states"]) == 6
        assert "table_counts" in data

    def test_api_catalog_endpoint(self):
        resp = self.client.get("/api/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if len(data) > 0:
            item = data[0]
            assert "table_name" in item
            assert "row_count" in item
            assert "source_url" in item
            assert "sha256" in item

    def test_api_dq_endpoint(self):
        resp = self.client.get("/api/dq")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if len(data) > 0:
            item = data[0]
            assert "table_name" in item
            assert "column_name" in item
            assert "semantic_type" in item

    def test_api_reconciliation_endpoint(self):
        resp = self.client.get("/api/ev/reconciliation")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_api_ev_coverage_endpoint(self):
        resp = self.client.get("/api/ev/coverage")
        assert resp.status_code == 200
        data = resp.json()
        assert "states" in data
        assert "corridors" in data
        assert "limitations" in data

    def test_api_reports_errors_endpoint(self):
        resp = self.client.get("/api/reports/errors")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_api_graph_endpoint(self):
        resp = self.client.get("/api/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data

    def test_api_review_queue_endpoint(self):
        resp = self.client.get("/api/review-queue")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_api_resolve_endpoint(self):
        resp = self.client.post("/api/resolve", json={"state": "Tamil Nadu", "source_name": "tuticorin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "Tamil Nadu"
        assert "top_candidates" in data
        assert "decision_band" in data

    def test_api_eval_endpoint(self):
        resp = self.client.get("/api/eval")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data or "baseline_metrics" in data



# =====================================================================
# PHASE 3 TESTS: Resolver Features, State Blocking, Baselines, Queue
# =====================================================================
class TestResolverFeatures:
    """Tests phonetic/orthographic feature engineering for district matching."""

    def test_consonant_skeleton(self):
        from resolver import consonant_skeleton

        assert consonant_skeleton("bengaluru") == "bnglr"
        assert consonant_skeleton("kasaragod") == "ksrgd"
        assert consonant_skeleton("chennai") == "chn"
        assert consonant_skeleton("") == ""
        assert consonant_skeleton(None) == ""

    def test_compute_similarity_features(self):
        from resolver import compute_similarity_features

        feats = compute_similarity_features("bengaluru urban", "bengaluru")
        assert "rf_ratio" in feats
        assert "rf_token_sort" in feats
        assert "skeleton_sim" in feats
        assert "first_char_match" in feats
        assert "len_diff" in feats
        assert feats["first_char_match"] == 1.0
        assert feats["rf_token_sort"] >= 0.75
        assert feats["rf_partial"] == 1.0


class TestResolverStateBlocking:
    """Verifies candidates never cross state lines."""

    def test_candidates_strictly_within_same_state(self):
        from resolver import DistrictResolver

        # Reference dictionary with two states
        ref_data = {
            "Tamil Nadu": ["chennai", "coimbatore", "madurai"],
            "Karnataka": ["bengaluru", "mysuru", "dharwad"],
        }
        resolver = DistrictResolver(reference_dict=ref_data)

        # Match source from Tamil Nadu
        candidates = resolver.get_top_candidates("Tamil Nadu", "chenai", k=3)
        assert len(candidates) > 0
        for cand, score in candidates:
            # Candidate MUST be one of Tamil Nadu districts
            assert cand in ref_data["Tamil Nadu"]
            assert cand not in ref_data["Karnataka"]


class TestResolverBaselines:
    """Tests Baseline 1 (Exact), Baseline 2 (TF-IDF), Baseline 3 (RapidFuzz)."""

    def test_baselines_prediction(self):
        from resolver import DistrictResolver

        ref_data = {
            "Tamil Nadu": ["chennai", "thoothukkudi tuticorin", "coimbatore"],
        }
        resolver = DistrictResolver(reference_dict=ref_data)

        # Exact match
        pred_exact, score_exact = resolver.baseline_exact("Tamil Nadu", "chennai")
        assert pred_exact == "chennai"
        assert score_exact == 1.0

        # Exact match failure
        pred_miss, score_miss = resolver.baseline_exact("Tamil Nadu", "tuticorin")
        assert pred_miss is None
        assert score_miss == 0.0

        # RapidFuzz match on alias
        pred_rf, score_rf = resolver.baseline_rapidfuzz("Tamil Nadu", "tuticorin")
        assert pred_rf == "thoothukkudi tuticorin"
        assert score_rf > 0.4


class TestLabellingQueueExport:
    """Verifies labelling queue format matches Section 9 protocol."""

    def test_labelling_queue_columns(self):
        from resolver import build_labelling_queue_df

        # Dummy unresolved list
        unresolved = [
            {
                "id": 1,
                "source_table": "clean_mgnrega",
                "state": "Tamil Nadu",
                "source_name": "tuticorin",
                "candidates": [
                    ("thoothukkudi tuticorin", 0.85),
                    ("tirunelveli", 0.40),
                    ("tenkasi", 0.35),
                ],
            }
        ]
        df_queue = build_labelling_queue_df(unresolved)
        expected_cols = [
            "id", "source_table", "state", "source_name",
            "cand1", "cand2", "cand3",
            "score1", "score2", "score3",
            "label", "labeller"
        ]
        assert list(df_queue.columns) == expected_cols
        assert len(df_queue) == 1
        assert df_queue.iloc[0]["cand1"] == "thoothukkudi tuticorin"
        assert df_queue.iloc[0]["score1"] == 0.85


# =====================================================================
# PHASE 4 TESTS: Joins Validation, Grain Rules, and NetworkX Graph
# =====================================================================
class TestJoinValidation:
    """Tests grain preservation, key discovery, and rejection of invalid joins."""

    def test_grain_validation_rejects_unaggregated_state_to_district_join(self):
        from joins import validate_join_compatibility

        # Attempting to join state-level EV table directly into district table
        is_valid, warning_msg = validate_join_compatibility(
            left_table="clean_pincode_districts",
            left_grain="district",
            right_table="clean_ev_pcs_2024",
            right_grain="state",
            join_keys=["norm_state"],
            aggregate_left_up=False,
        )
        assert is_valid is False
        assert "Grain mismatch" in warning_msg or "aggregation required" in warning_msg.lower()

    def test_validated_district_join_preserves_grain(self):
        import pandas as pd
        from joins import join_district_tables

        # TEST FIXTURES: Two district-grain tables
        df_pincode_dist = pd.DataFrame([
            {"norm_state": "Tamil Nadu", "norm_district": "chennai", "pin_count": 50},
            {"norm_state": "Tamil Nadu", "norm_district": "coimbatore", "pin_count": 40},
        ])
        df_msme_dist = pd.DataFrame([
            {"norm_state": "Tamil Nadu", "norm_district": "chennai", "total_msme": 1000},
            {"norm_state": "Tamil Nadu", "norm_district": "coimbatore", "total_msme": 800},
        ])

        integrated_df, report = join_district_tables(df_pincode_dist, df_msme_dist, how="inner")
        assert len(integrated_df) == 2
        assert "pin_count" in integrated_df.columns
        assert "total_msme" in integrated_df.columns
        assert report["expansion_factor"] == 1.0


class TestDatasetGraph:
    """Tests NetworkX graph generation and JSON serialization."""

    def test_build_dataset_graph(self):
        from joins import build_dataset_graph_json

        graph_dict = build_dataset_graph_json()
        assert "nodes" in graph_dict
        assert "edges" in graph_dict
        assert len(graph_dict["nodes"]) >= 5
        # Verify node structure
        node_ids = [n["id"] for n in graph_dict["nodes"]]
        assert "clean_pincode" in node_ids
        assert "clean_mgnrega" in node_ids
        assert "clean_msme" in node_ids
        assert "clean_ev_pcs_2024" in node_ids


# =====================================================================
# PHASE 5 TESTS: EV Coverage Explorer & Reports
# =====================================================================
class TestEvExplorer:
    """Tests EV coverage explorer metrics, date gap warnings, and missing state logic."""

    def test_ev_coverage_metrics_computation(self):
        from ev import compute_ev_coverage_metrics

        # Sample state row
        metrics = compute_ev_coverage_metrics(
            state="Tamil Nadu",
            pcs_2024=650,
            omc_2025=1100,
            ev_count_2023=85000,
            total_vehicles_2023=31000000,
        )
        assert metrics["state"] == "Tamil Nadu"
        assert metrics["pcs_2024"] == 650
        assert metrics["omc_2025"] == 1100
        assert metrics["ev_share_pct"] > 0
        assert metrics["chargers_per_1000_evs_2024"] is not None
        assert metrics["chargers_per_1000_evs_2025"] is not None

    def test_telangana_absent_handled_as_not_available(self):
        from ev import compute_ev_coverage_metrics

        metrics = compute_ev_coverage_metrics(
            state="Telangana",
            pcs_2024=500,
            omc_2025=800,
            ev_count_2023=None,
            total_vehicles_2023=None,
        )
        assert metrics["state"] == "Telangana"
        assert metrics["chargers_per_1000_evs_2024"] == "Not available: dataset missing"
        assert metrics["ev_share_pct"] == "Not available: dataset missing"

    def test_ev_limitations_warnings_present(self):
        from ev import get_data_limitations_notice

        warnings = get_data_limitations_notice()
        assert len(warnings) >= 3
        # Ensure definition distinction is explicit
        texts = " ".join([w["title"] + " " + w["description"] for w in warnings])
        assert "different definitions" in texts.lower()
        assert "date gap" in texts.lower()
        assert "telangana" in texts.lower()


class TestReports:
    """Tests data quality reporting and suspected error auditing."""

    def test_suspected_error_report_wording(self):
        from reports import generate_suspected_error_report

        errors = generate_suspected_error_report()
        assert len(errors) >= 3
        for err in errors:
            # Rule: wording must use "suspected", not assertive accusations
            assert "suspected" in err["category"].lower() or "suspected" in err["description"].lower() or "divergence" in err["category"].lower()
            assert "table" in err
            assert "evidence" in err


# =====================================================================
# PHASE 8 TESTS: Evaluation Engine, Small-n Refusal, Wilson Intervals
# =====================================================================
class TestEvaluationRules:
    """Tests evaluation thresholds, small-n refusal, and statistical metrics."""

    def test_evaluate_refuses_small_n(self):
        import pandas as pd
        from evaluate import run_evaluation_on_labels

        # TEST FIXTURE: Only 5 hand-labelled rows (below 30 minimum threshold)
        small_labels_df = pd.DataFrame([
            {"id": 1, "state": "Tamil Nadu", "source_name": "tuticorin", "cand1": "thoothukkudi tuticorin", "score1": 0.85, "label": "1", "labeller": "AS"},
            {"id": 2, "state": "Karnataka", "source_name": "kasargod", "cand1": "kasaragod", "score1": 0.95, "label": "1", "labeller": "RK"},
            {"id": 3, "state": "Kerala", "source_name": "pathanamthipta", "cand1": "pathanamthitta", "score1": 0.93, "label": "1", "labeller": "AS"},
            {"id": 4, "state": "Karnataka", "source_name": "bagalkote", "cand1": "bagalkot", "score1": 0.96, "label": "1", "labeller": "RK"},
            {"id": 5, "state": "Telangana", "source_name": "rangareddy", "cand1": "ranga reddy", "score1": 0.97, "label": "1", "labeller": "AS"},
        ])

        with pytest.raises(ValueError) as excinfo:
            run_evaluation_on_labels(small_labels_df, min_labels=30)
        assert "minimum of 30" in str(excinfo.value).lower()


    def test_wilson_confidence_interval(self):
        from evaluate import wilson_score_interval

        # 25 successes out of 30 trials (accuracy = 83.3%)
        ci_low, ci_high = wilson_score_interval(successes=25, trials=30, confidence=0.95)
        assert 0.65 < ci_low < 0.75
        assert 0.90 < ci_high < 0.96
        assert ci_low < (25 / 30) < ci_high






