"""
South India Open Data Integration Engine - Ingestion and Cleaning Pipeline
Transforms raw cached data.gov.in JSON files into validated, clean DuckDB tables.
Layers:
  1. raw_*     : Unmodified source payloads as extracted from portal JSON.
  2. staging_* : Parsed types (strings -> integers, floats, plausibility flags).
  3. clean_*   : Target 6-state filtered, deduplicated, reconciled datasets.
  4. meta_*    : Lineage, transformation logs, rejected records, reconciliation, and DQ profiling.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from unidecode import unidecode

from config import DATASETS, RAW_DIR, TARGET_STATES

logger = logging.getLogger(__name__)

# Standard 6 target South Indian states in title case
TARGET_SOUTH_STATES = [
    "Tamil Nadu",
    "Karnataka",
    "Kerala",
    "Andhra Pradesh",
    "Telangana",
    "Puducherry",
]

# Observed state name variations across official data.gov.in tables
# Built STRICTLY from variants actually present in the portal files
STATE_NORMALISATION_MAP: Dict[str, str] = {
    "tamil nadu": "Tamil Nadu",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "andhra pradesh": "Andhra Pradesh",
    "telangana": "Telangana",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "andaman and nicobar island": "Andaman and Nicobar Islands",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "jammu and kashmir": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "delhi": "Delhi",
    "chandigarh": "Chandigarh",
    "lakshadweep": "Lakshadweep",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "maharashtra": "Maharashtra",
    "rajasthan": "Rajasthan",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttar pradesh": "Uttar Pradesh",
    "bihar": "Bihar",
    "jharkhand": "Jharkhand",
    "west bengal": "West Bengal",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "chhattisgarh": "Chhattisgarh",
    "madhya pradesh": "Madhya Pradesh",
    "assam": "Assam",
    "sikkim": "Sikkim",
    "arunachal pradesh": "Arunachal Pradesh",
    "nagaland": "Nagaland",
    "manipur": "Manipur",
    "mizoram": "Mizoram",
    "tripura": "Tripura",
    "meghalaya": "Meghalaya",
}


def normalize_state_name(state_raw: Optional[str]) -> Optional[str]:
    """
    Standardise state and UT names using observed variations across ministries.
    Returns Title-cased official standard name, or None if invalid/empty.
    """
    if state_raw is None:
        return None
    s = str(state_raw).strip()
    if not s or s.lower() in {"na", "null", "none", ""}:
        return None
    cleaned = re.sub(r"\s+", " ", s).lower()
    if cleaned in STATE_NORMALISATION_MAP:
        return STATE_NORMALISATION_MAP[cleaned]
    # Fallback to standard title case for unexpected valid states
    return s.title()


def normalize_district_name(district_raw: Optional[str]) -> Optional[str]:
    """
    Clean district strings for baseline and ML matching.
    Removes accents, punctuation, parentheses, trailing 'district', leading 'the'.
    """
    if district_raw is None:
        return None
    s = str(district_raw).strip()
    if not s or s.lower() in {"na", "null", "none", "", "nan"}:
        return None

    # Convert accents (e.g. Kozhikode variants) to ASCII
    s = unidecode(s).lower()
    # Replace parentheses and commas with spaces: e.g. "Thoothukkudi (Tuticorin)" -> "thoothukkudi tuticorin"
    s = re.sub(r"[\(\)\[\],\.\/\\_-]", " ", s)
    # Strip standalone words like 'district' or leading 'the'
    tokens = s.split()
    filtered_tokens = [
        t for t in tokens if t not in {"district", "dist"}
    ]
    if filtered_tokens and filtered_tokens[0] == "the":
        filtered_tokens = filtered_tokens[1:]
    res = " ".join(filtered_tokens).strip()
    return res if res else None


def clean_pincode(pin_val: Any) -> Optional[str]:
    """
    Validate 6-digit Indian Postal Index Number.
    Indian PINs are 6 digits, where first digit represents region (1-9; never 0).
    """
    if pin_val is None:
        return None
    s = str(pin_val).replace(" ", "").replace("-", "").strip()
    if len(s) == 6 and s.isdigit() and s[0] != "0":
        return s
    return None


def clean_coordinate(val: Any, coord_type: str = "lat") -> Optional[float]:
    """
    Validate geographic coordinates for plausibility in India.
    Filters out zeros, 'NA', and values outside India's bounding box.
    India latitude bounding: ~6.0 N to 38.0 N.
    India longitude bounding: ~68.0 E to 98.0 E.
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"", "0", "0.0", "na", "null", "none", "nan"}:
        return None
    try:
        f = float(s)
    except (ValueError, TypeError):
        return None

    if coord_type == "lat":
        if 6.0 <= f <= 38.0:
            return round(f, 6)
    elif coord_type == "lon":
        if 68.0 <= f <= 98.0:
            return round(f, 6)
    return None


def separate_summary_row(
    df: pd.DataFrame,
    state_col: str,
    val_col: str,
    summary_pattern: str = "Total",
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Identify and extract aggregate summary rows inside state-level tables.
    Compares the portal-reported total against sum of individual state rows.
    """
    if df.empty or state_col not in df.columns:
        return df, None

    mask = df[state_col].astype(str).str.contains(summary_pattern, case=False, na=False)
    summary_rows = df[mask]
    clean_df = df[~mask].copy()

    if summary_rows.empty:
        return clean_df, None

    # Parse numeric total from first summary row
    summary_row = summary_rows.iloc[0]
    reported_str = str(summary_row[val_col]).replace(",", "").strip()
    try:
        reported_val = float(reported_str)
    except (ValueError, TypeError):
        reported_val = 0.0

    # Calculate sum of individual state rows
    def _parse_num(x: Any) -> float:
        try:
            return float(str(x).replace(",", "").strip())
        except (ValueError, TypeError):
            return 0.0

    calculated_sum = clean_df[val_col].apply(_parse_num).sum()
    diff = round(calculated_sum - reported_val, 4)

    reconciliation_info = {
        "reported_total": reported_val,
        "calculated_sum": calculated_sum,
        "difference": diff,
        "status": "RECONCILED" if diff == 0 else f"DISCREPANCY_{diff}",
    }
    return clean_df, reconciliation_info


def infer_semantic_type(col_name: str, sample_series: pd.Series) -> str:
    """Classify a column into a domain semantic type for data profiling."""
    lower_name = col_name.lower()
    if "pincode" in lower_name or "pin" in lower_name:
        return "pincode"
    if "latitude" in lower_name or lower_name in {"lat", "latitude"}:
        return "coordinate_latitude"
    if "longitude" in lower_name or lower_name in {"lon", "long", "longitude"}:
        return "coordinate_longitude"
    if "state" in lower_name:
        return "state_name"
    if "district" in lower_name:
        return "district_name"
    if any(k in lower_name for k in ["count", "total", "num", "applied", "pcs", "station", "vehicle"]):
        return "metric_count"
    if any(k in lower_name for k in ["year", "date", "month"]):
        return "temporal_date"
    return "text_attribute"


class DataPipeline:
    """
    Main ingestion, cleaning, and profiling engine.
    Persists data in DuckDB with strictly separated raw, staging, clean, and meta schemas.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is None or str(db_path) == ":memory:":
            self.conn = duckdb.connect(":memory:")
            self.db_path = None
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = duckdb.connect(str(self.db_path))
        self._init_metadata_tables()

    def _init_metadata_tables(self) -> None:
        """Create lineage, transformation log, rejection, and profiling tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_lineage (
                table_name VARCHAR,
                source_url VARCHAR,
                resource_id VARCHAR,
                sha256 VARCHAR,
                row_count INTEGER,
                fetch_date VARCHAR,
                created_at TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_transform_log (
                step VARCHAR,
                table_name VARCHAR,
                description VARCHAR,
                rows_in INTEGER,
                rows_out INTEGER,
                timestamp TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_rejected_records (
                table_name VARCHAR,
                reason VARCHAR,
                raw_record_json VARCHAR,
                timestamp TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_ev_reconciliation (
                table_name VARCHAR,
                reported_total DOUBLE,
                calculated_sum DOUBLE,
                difference DOUBLE,
                status VARCHAR,
                timestamp TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_data_quality (
                table_name VARCHAR,
                column_name VARCHAR,
                semantic_type VARCHAR,
                total_rows INTEGER,
                null_count INTEGER,
                unique_count INTEGER,
                notes VARCHAR
            );
        """)

    def log_transform(
        self, step: str, table_name: str, desc: str, rows_in: int, rows_out: int
    ) -> None:
        now = datetime.now(timezone.utc)
        self.conn.execute(
            "INSERT INTO meta_transform_log VALUES (?, ?, ?, ?, ?, ?)",
            [step, table_name, desc, rows_in, rows_out, now],
        )

    def log_rejection(self, table_name: str, reason: str, record_str: str) -> None:
        now = datetime.now(timezone.utc)
        self.conn.execute(
            "INSERT INTO meta_rejected_records VALUES (?, ?, ?, ?)",
            [table_name, reason, record_str, now],
        )

    def get_table_names(self) -> List[str]:
        res = self.conn.execute("SHOW TABLES").fetchall()
        return [r[0] for r in res]

    def close(self) -> None:
        self.conn.close()

    def run(self, raw_dir: Optional[Path] = None) -> None:
        """
        Execute the entire data ingestion and cleaning pipeline.
        Reads all 7 verified portal JSON files and constructs clean tables in DuckDB.
        """
        data_dir = Path(raw_dir) if raw_dir else RAW_DIR
        manifest_file = data_dir / "manifest.json"
        manifest = {}
        if manifest_file.exists():
            with open(manifest_file, "r") as f:
                manifest = json.load(f)

        now = datetime.now(timezone.utc)

        # -------------------------------------------------------------
        # 1. PINCODE DIRECTORY (Department of Posts)
        # -------------------------------------------------------------
        pin_file = data_dir / "pincode_directory.json"
        if pin_file.exists():
            with open(pin_file, "r") as f:
                pin_raw = json.load(f)
            df_pin = pd.DataFrame(pin_raw)
            self.conn.register("df_pin_raw", df_pin)
            self.conn.execute("CREATE OR REPLACE TABLE raw_pincode AS SELECT * FROM df_pin_raw")
            self.conn.unregister("df_pin_raw")

            rows_in = len(df_pin)
            # Normalise state
            df_pin["norm_state"] = df_pin["statename"].apply(normalize_state_name)
            df_pin["norm_district"] = df_pin["district"].apply(normalize_district_name)
            df_pin["clean_pin"] = df_pin["pincode"].apply(clean_pincode)
            df_pin["clean_lat"] = df_pin["latitude"].apply(lambda v: clean_coordinate(v, "lat"))
            df_pin["clean_lon"] = df_pin["longitude"].apply(lambda v: clean_coordinate(v, "lon"))

            # Filter to 6 target South Indian states
            mask_target = df_pin["norm_state"].isin(TARGET_SOUTH_STATES)
            excluded_count = int((~mask_target).sum())
            df_pin_clean = df_pin[mask_target].copy()

            self.conn.register("df_pin_clean_reg", df_pin_clean)
            self.conn.execute("CREATE OR REPLACE TABLE clean_pincode AS SELECT * FROM df_pin_clean_reg")
            self.conn.unregister("df_pin_clean_reg")

            self.log_transform(
                step="filter_south_india",
                table_name="clean_pincode",
                desc=f"Filtered {rows_in} nationwide rows to 6 target South Indian states (excluded {excluded_count} rows from UP, MP, HP, AR)",
                rows_in=rows_in,
                rows_out=len(df_pin_clean),
            )
            # Record lineage
            meta = manifest.get("pincode_directory", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_pincode",
                    meta.get("url", "https://www.data.gov.in/resource/all-india-pincode-directory-till-last-month"),
                    meta.get("id", "6176ee09-3d56-4a3b-8115-21841576b2f6"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_pin_clean),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # 2. MGNREGA DISTRICT MONTHLY PANEL
        # -------------------------------------------------------------
        mgnrega_file = data_dir / "mgnrega_district.json"
        if mgnrega_file.exists():
            with open(mgnrega_file, "r") as f:
                mg_raw = json.load(f)
            df_mg = pd.DataFrame(mg_raw)
            self.conn.register("df_mg_raw", df_mg)
            self.conn.execute("CREATE OR REPLACE TABLE raw_mgnrega AS SELECT * FROM df_mg_raw")
            self.conn.unregister("df_mg_raw")

            rows_in = len(df_mg)
            df_mg["norm_state"] = df_mg["state_name"].apply(normalize_state_name)
            df_mg["norm_district"] = df_mg["district_name"].apply(normalize_district_name)

            # Cast key numerical metrics
            numeric_cols = [
                "Persondays_of_Central_Liability_so_far",
                "Total_Exp",
                "Total_Households_Worked",
                "Total_Individuals_Worked",
                "Total_No_of_Active_Job_Cards",
                "Total_No_of_Active_Workers",
            ]
            for col in numeric_cols:
                if col in df_mg.columns:
                    df_mg[col] = pd.to_numeric(df_mg[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

            # Documented Deduplication Rule:
            # MGNREGA rows represent monthly cumulative snapshots across financial years.
            # Multiple records for the same (state_name, district_name, fin_year, month) appear due to cumulative revisions.
            # Rule: Keep the record with the maximum persondays or expenditure, ensuring no double counting.
            sort_cols = [c for c in ["Persondays_of_Central_Liability_so_far", "Total_Exp"] if c in df_mg.columns]
            if sort_cols:
                df_mg_sorted = df_mg.sort_values(by=sort_cols, ascending=False)
            else:
                df_mg_sorted = df_mg

            df_mg_clean = df_mg_sorted.drop_duplicates(
                subset=["norm_state", "norm_district", "fin_year", "month"],
                keep="first",
            ).copy()


            self.conn.register("df_mg_clean_reg", df_mg_clean)
            self.conn.execute("CREATE OR REPLACE TABLE clean_mgnrega AS SELECT * FROM df_mg_clean_reg")
            self.conn.unregister("df_mg_clean_reg")

            dedup_dropped = rows_in - len(df_mg_clean)
            self.log_transform(
                step="mgnrega_panel_deduplication",
                table_name="clean_mgnrega",
                desc=f"Deduplicated monthly panel across (state, district, fin_year, month), retaining latest cumulative revision. Suppressed {dedup_dropped} duplicate snapshots.",
                rows_in=rows_in,
                rows_out=len(df_mg_clean),
            )
            meta = manifest.get("mgnrega_district", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_mgnrega",
                    meta.get("url", "https://www.data.gov.in/resource/district-wise-monthly-employment-generated-under-mgnrega"),
                    meta.get("id", "eeab5eb3-9764-4286-9dc5-dcae8ca47195"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_mg_clean),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # 3. MSME UDYAM DISTRICT DATA
        # -------------------------------------------------------------
        msme_file = data_dir / "msme_district.json"
        if msme_file.exists():
            with open(msme_file, "r") as f:
                msme_raw = json.load(f)
            df_msme = pd.DataFrame(msme_raw)
            self.conn.register("df_msme_raw", df_msme)
            self.conn.execute("CREATE OR REPLACE TABLE raw_msme AS SELECT * FROM df_msme_raw")
            self.conn.unregister("df_msme_raw")

            rows_in = len(df_msme)
            df_msme["norm_state"] = df_msme["statename"].apply(normalize_state_name)
            df_msme["norm_district"] = df_msme["districtname"].apply(normalize_district_name)

            for col in ["totalnoofapplication", "totalnoof_micro", "totalnoof_small", "totalnoof_medium"]:
                if col in df_msme.columns:
                    df_msme[col] = pd.to_numeric(df_msme[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

            # Filter to target 6 states
            mask_msme_target = df_msme["norm_state"].isin(TARGET_SOUTH_STATES)
            df_msme_clean = df_msme[mask_msme_target].copy()
            excluded_msme = rows_in - len(df_msme_clean)

            self.conn.register("df_msme_clean_reg", df_msme_clean)
            self.conn.execute("CREATE OR REPLACE TABLE clean_msme AS SELECT * FROM df_msme_clean_reg")
            self.conn.unregister("df_msme_clean_reg")

            self.log_transform(
                step="filter_south_india",
                table_name="clean_msme",
                desc=f"Filtered {rows_in} nationwide MSME districts to 6 South Indian territories (retained {len(df_msme_clean)}, excluded {excluded_msme})",
                rows_in=rows_in,
                rows_out=len(df_msme_clean),
            )
            meta = manifest.get("msme_district", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_msme",
                    meta.get("url", "https://www.data.gov.in/resource/state-and-district-wise-number-enterprises-applied-under-udyam-registration"),
                    meta.get("id", "c06497f2-1b1f-49ff-813c-4ef1e0d45362"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_msme_clean),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # 4. EV CHARGING STATIONS 2024 (Operational PCS)
        # -------------------------------------------------------------
        ev24_file = data_dir / "ev_charging_state_2024.json"
        if ev24_file.exists():
            with open(ev24_file, "r") as f:
                ev24_raw = json.load(f)
            df_ev24 = pd.DataFrame(ev24_raw)
            self.conn.register("df_ev24_raw", df_ev24)
            self.conn.execute("CREATE OR REPLACE TABLE raw_ev_pcs_2024 AS SELECT * FROM df_ev24_raw")
            self.conn.unregister("df_ev24_raw")

            state_col = "state_ut" if "state_ut" in df_ev24.columns else df_ev24.columns[1]
            pcs_col = [c for c in df_ev24.columns if "pcs" in c.lower() or "operational" in c.lower()][0]

            df_ev24_clean, rec_info = separate_summary_row(df_ev24, state_col, pcs_col, "Total")
            df_ev24_clean["norm_state"] = df_ev24_clean[state_col].apply(normalize_state_name)
            df_ev24_clean["operational_pcs"] = pd.to_numeric(
                df_ev24_clean[pcs_col].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)

            self.conn.register("df_ev24_reg", df_ev24_clean)
            self.conn.execute("CREATE OR REPLACE TABLE clean_ev_pcs_2024 AS SELECT * FROM df_ev24_reg")
            self.conn.unregister("df_ev24_reg")

            if rec_info:
                self.conn.execute(
                    "INSERT INTO meta_ev_reconciliation VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        "clean_ev_pcs_2024",
                        rec_info["reported_total"],
                        rec_info["calculated_sum"],
                        rec_info["difference"],
                        rec_info["status"],
                        now,
                    ],
                )

            meta = manifest.get("ev_charging_state_2024", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_ev_pcs_2024",
                    meta.get("url", "https://www.data.gov.in/resource/stateut-wise-details-operational-public-charging-stations-pcs-31032024"),
                    meta.get("id", "1d977457-36e4-4d8b-bb8c-7f5fbca41c1e"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_ev24_clean),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # 5. EV CHARGING STATIONS 2025 (PSU OMC)
        # -------------------------------------------------------------
        ev25_file = data_dir / "ev_charging_omc_2025.json"
        if ev25_file.exists():
            with open(ev25_file, "r") as f:
                ev25_raw = json.load(f)
            df_ev25 = pd.DataFrame(ev25_raw)
            self.conn.register("df_ev25_raw", df_ev25)
            self.conn.execute("CREATE OR REPLACE TABLE raw_ev_omc_2025 AS SELECT * FROM df_ev25_raw")
            self.conn.unregister("df_ev25_raw")

            state_col = "state_ut" if "state_ut" in df_ev25.columns else df_ev25.columns[1]
            tot_col = [c for c in df_ev25.columns if "total" in c.lower()][-1]

            df_ev25_clean, rec_info = separate_summary_row(df_ev25, state_col, tot_col, "Total")
            df_ev25_clean["norm_state"] = df_ev25_clean[state_col].apply(normalize_state_name)
            df_ev25_clean["omc_total_evcs"] = pd.to_numeric(
                df_ev25_clean[tot_col].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)

            self.conn.register("df_ev25_reg", df_ev25_clean)
            self.conn.execute("CREATE OR REPLACE TABLE clean_ev_omc_2025 AS SELECT * FROM df_ev25_reg")
            self.conn.unregister("df_ev25_reg")

            if rec_info:
                self.conn.execute(
                    "INSERT INTO meta_ev_reconciliation VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        "clean_ev_omc_2025",
                        rec_info["reported_total"],
                        rec_info["calculated_sum"],
                        rec_info["difference"],
                        rec_info["status"],
                        now,
                    ],
                )

            meta = manifest.get("ev_charging_omc_2025", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_ev_omc_2025",
                    meta.get("url", "https://www.data.gov.in/resource/state-wise-details-ev-charging-stations-installedenergized-oil-marketing-companies-01012025"),
                    meta.get("id", "3bc8565a-be18-472e-84dd-2436ba45ec77"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_ev25_clean),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # 6. REGISTERED VEHICLES 2023 (e-Vahan)
        # -------------------------------------------------------------
        vahan_file = data_dir / "ev_registered_vehicles_2023.json"
        if vahan_file.exists():
            with open(vahan_file, "r") as f:
                vahan_raw = json.load(f)
            df_vahan = pd.DataFrame(vahan_raw)
            self.conn.register("df_vahan_raw", df_vahan)
            self.conn.execute("CREATE OR REPLACE TABLE raw_ev_vahan_2023 AS SELECT * FROM df_vahan_raw")
            self.conn.unregister("df_vahan_raw")

            state_col = "state_ut" if "state_ut" in df_vahan.columns else df_vahan.columns[1]
            ev_col = [c for c in df_vahan.columns if "electric" in c.lower()][0]
            tot_col = [c for c in df_vahan.columns if "total" in c.lower()][0]

            df_vahan_clean, rec_info = separate_summary_row(df_vahan, state_col, tot_col, "Grand Total")
            df_vahan_clean["norm_state"] = df_vahan_clean[state_col].apply(normalize_state_name)
            df_vahan_clean["ev_count"] = pd.to_numeric(
                df_vahan_clean[ev_col].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)
            df_vahan_clean["total_vehicles"] = pd.to_numeric(
                df_vahan_clean[tot_col].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)

            self.conn.register("df_vahan_reg", df_vahan_clean)
            self.conn.execute("CREATE OR REPLACE TABLE clean_ev_vahan_2023 AS SELECT * FROM df_vahan_reg")
            self.conn.unregister("df_vahan_reg")

            if rec_info:
                self.conn.execute(
                    "INSERT INTO meta_ev_reconciliation VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        "clean_ev_vahan_2023",
                        rec_info["reported_total"],
                        rec_info["calculated_sum"],
                        rec_info["difference"],
                        rec_info["status"],
                        now,
                    ],
                )

            meta = manifest.get("ev_registered_vehicles_2023", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_ev_vahan_2023",
                    meta.get("url", "https://www.data.gov.in/resource/state-wise-data-electric-vehicles-and-total-vehicles-registered-14072023"),
                    meta.get("id", "62402127-b649-411a-8280-993d052cb443"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_vahan_clean),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # 7. HIGHWAY CORRIDORS 2022
        # -------------------------------------------------------------
        corridor_file = data_dir / "ev_city_highway_2022.json"
        if corridor_file.exists():
            with open(corridor_file, "r") as f:
                corr_raw = json.load(f)
            df_corr = pd.DataFrame(corr_raw)
            self.conn.register("df_corr_raw", df_corr)
            self.conn.execute("CREATE OR REPLACE TABLE raw_ev_corridor_2022 AS SELECT * FROM df_corr_raw")
            self.conn.unregister("df_corr_raw")

            val_col = [c for c in df_corr.columns if "charging" in c.lower() or "stations" in c.lower()][0]
            name_col = [c for c in df_corr.columns if "highway" in c.lower() or "expressway" in c.lower() or "location" in c.lower()][0]

            df_corr["station_count"] = pd.to_numeric(
                df_corr[val_col].astype(str).str.replace(",", ""), errors="coerce"
            ).fillna(0)
            df_corr["corridor_name"] = df_corr[name_col].astype(str).str.strip()

            self.conn.register("df_corr_reg", df_corr)
            self.conn.execute("CREATE OR REPLACE TABLE clean_ev_corridor_2022 AS SELECT * FROM df_corr_reg")
            self.conn.unregister("df_corr_reg")

            meta = manifest.get("ev_city_highway_2022", {})
            self.conn.execute(
                "INSERT INTO meta_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    "clean_ev_corridor_2022",
                    meta.get("url", "https://www.data.gov.in/resource/details-operational-charging-stations-various-expressway-and-highways-across-country-07122022"),
                    meta.get("id", "93ca4b4c-9f79-43c2-a42e-1311b93ea300"),
                    meta.get("sha256", "NOT VERIFIED"),
                    len(df_corr),
                    meta.get("fetch_date", "NOT VERIFIED"),
                    now,
                ],
            )

        # -------------------------------------------------------------
        # DATA QUALITY PROFILING
        # -------------------------------------------------------------
        self.profile_all_tables()

    def profile_all_tables(self) -> None:
        """Scan clean tables and generate data quality metrics for each column."""
        self.conn.execute("DELETE FROM meta_data_quality")
        clean_tables = [t for t in self.get_table_names() if t.startswith("clean_")]
        for tbl in clean_tables:
            df = self.conn.execute(f"SELECT * FROM {tbl}").fetchdf()
            total_rows = len(df)
            for col in df.columns:
                null_cnt = int(df[col].isna().sum())
                unique_cnt = int(df[col].nunique(dropna=True))
                sem_type = infer_semantic_type(col, df[col])
                notes = ""
                if null_cnt > 0:
                    pct = round((null_cnt / total_rows) * 100, 1)
                    notes = f"{pct}% nulls"
                self.conn.execute(
                    "INSERT INTO meta_data_quality VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [tbl, col, sem_type, total_rows, null_cnt, unique_cnt, notes],
                )


def run_pipeline(
    raw_dir: Optional[Path] = None,
    db_path: Optional[Path | str] = None,
) -> DataPipeline:
    """Convenience entry point to run the pipeline."""
    target_db = db_path if db_path is not None else Path("data/work/engine.duckdb")
    pipeline = DataPipeline(db_path=target_db)
    pipeline.run(raw_dir=raw_dir)
    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run South India Open Data Ingestion Pipeline")
    parser.add_argument("--db", type=str, default="data/work/engine.duckdb", help="DuckDB file path")
    args = parser.parse_args()

    print(f"[*] Starting ingestion pipeline into {args.db}...")
    pipe = run_pipeline(db_path=args.db)
    print(f"[✓] Pipeline complete. Generated tables: {pipe.get_table_names()}")
    pipe.close()
