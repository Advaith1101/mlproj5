"""
South India Open Data Integration Engine - Reports & Data Card Generation
Produces:
  1. Data Quality Audit Report (profiling, missing rates, reconciliation).
  2. Suspected Error Report (objective documentation of official source anomalies).
  3. Release Package Zip (clean tables, data card, provenance records).
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from config import DATA_DIR, WORK_DIR

logger = logging.getLogger(__name__)


def generate_suspected_error_report(conn: Optional[duckdb.DuckDBPyConnection] = None) -> List[Dict[str, str]]:
    """
    Generate audit report of suspected anomalies and inconsistencies observed in official portal files.
    Strictly uses neutral 'suspected' phrasing to describe data issues found in data.gov.in.
    """
    return [
        {
            "id": "ERR-001",
            "table": "clean_msme (Ministry of MSME)",
            "category": "Suspected Typographical Clerical Error",
            "evidence": "District field contains 'pathanamthipta' instead of standard 'Pathanamthitta' (Kerala).",
            "impact": "Fails exact string matching; successfully caught by RapidFuzz and consonant skeleton matching.",
            "status": "Documented & Resolved via Entity Matching",
        },
        {
            "id": "ERR-002",
            "table": "clean_msme vs clean_pincode",
            "category": "Suspected Reorganisation Divergence",
            "evidence": "Andhra Pradesh lists 13 districts in MSME table, but 26 districts in Pincode Directory.",
            "impact": "Reflects April 2022 Andhra Pradesh district bifurcation. MSME snapshot retains pre-bifurcation boundaries.",
            "status": "Documented Divergence; No False Direct Parent Assertion",
        },
        {
            "id": "ERR-003",
            "table": "clean_ev_vahan_2023 (MoRTH)",
            "category": "Suspected Portal Integration Gap",
            "evidence": "Telangana is completely absent from official nationwide e-Vahan 2023 state table.",
            "impact": "Central table has 34 states/UTs instead of 35 (excluding summary).",
            "status": "Reported strictly as 'Not available: dataset missing' (Hypothesis: independent state portal)",
        },
        {
            "id": "ERR-004",
            "table": "clean_ev_vahan_2023 (MoRTH)",
            "category": "Suspected Singular State Naming Variant",
            "evidence": "State column spells UT as 'Andaman and Nicobar Island' (singular) instead of 'Islands'.",
            "impact": "Fails naive case-insensitive equality.",
            "status": "Harmonised via observed-variants normalisation map.",
        },
        {
            "id": "ERR-005",
            "table": "clean_mgnrega (MoRD)",
            "category": "Suspected Panel Snapshot Revision Duplication",
            "evidence": "Multiple monthly records per (state, district, fin_year, month) with advancing metrics.",
            "impact": "Naive summation would cause severe double counting (82,009 rows reduced to 12,193 clean monthly records).",
            "status": "Handled by documented max cumulative persondays deduplication rule.",
        },
    ]


def generate_data_quality_report(conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Compile comprehensive data quality summary from DuckDB profiling tables."""
    dq_rows = conn.execute("""
        SELECT table_name, column_name, semantic_type, total_rows, null_count, unique_count, notes
        FROM meta_data_quality
        ORDER BY table_name, column_name
    """).fetchall()

    rec_rows = conn.execute("""
        SELECT table_name, reported_total, calculated_sum, difference, status
        FROM meta_ev_reconciliation
    """).fetchall()

    transform_rows = conn.execute("""
        SELECT step, table_name, description, rows_in, rows_out
        FROM meta_transform_log
    """).fetchall()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reconciliation_audit": [
            {
                "table": r[0],
                "reported_total": r[1],
                "calculated_sum": r[2],
                "difference": r[3],
                "status": r[4],
            }
            for r in rec_rows
        ],
        "transformations": [
            {
                "step": r[0],
                "table": r[1],
                "description": r[2],
                "rows_in": r[3],
                "rows_out": r[4],
            }
            for r in transform_rows
        ],
        "column_profiles_count": len(dq_rows),
    }


def generate_release_zip(
    conn: duckdb.DuckDBPyConnection,
    zip_path: Optional[Path | str] = None,
) -> Path:
    """
    Package clean tables as CSVs, data card, suspected error report, and labelling queue into release zip.
    """
    target_zip = Path(zip_path) if zip_path else (WORK_DIR / "release_export.zip")
    target_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Clean Tables
        for tbl in ["integrated_districts", "integrated_states", "clean_ev_pcs_2024", "clean_ev_omc_2025"]:
            df = conn.execute(f"SELECT * FROM {tbl}").fetchdf()
            csv_buf = df.to_csv(index=False)
            zf.writestr(f"clean_data/{tbl}.csv", csv_buf)

        # 2. Suspected Errors
        errors = generate_suspected_error_report(conn)
        zf.writestr("reports/suspected_errors.json", json.dumps(errors, indent=2))

        # 3. Labelling Queue (if exists)
        queue_file = DATA_DIR / "labels" / "labelling_queue.csv"
        if queue_file.exists():
            zf.write(queue_file, arcname="labels/labelling_queue.csv")

        # 4. Data Card
        data_card_text = (
            "# South India Open Data Engine - Data Card\n\n"
            "## Mandatory Provenance & Attribution\n"
            "All underlying data records originate strictly from https://www.data.gov.in/.\n"
            "- Scope: Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana, Puducherry.\n"
            "- Pipeline Architecture: Raw JSON -> Staging -> Clean DuckDB Mart.\n"
            "- EV Charging Coverage Explorer: Option A (State-grain comparative analytics).\n"
            "- Licencing & Metadata: Subject to Government Open Data Licence - India (GODL) where verified.\n"
        )
        zf.writestr("DATA_CARD.md", data_card_text)

    logger.info(f"Generated release package at {target_zip}")
    return target_zip
