"""
South India Open Data Integration Engine - EV Charging Coverage Explorer
State-grain comparative analytics across verified government tables.
Rules:
  - PCS (2024) and OMC (2025) have DIFFERENT definitions; NEVER combine as a time series.
  - Chargers per 1,000 EVs uses 2023 e-Vahan; must include visible date-gap warning.
  - Telangana is absent from the 2023 e-Vahan table; report 'Not available: dataset missing'.
  - No coordinates, no trip planner, no trust scores, no claims of shortage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

from config import TARGET_STATES, WORK_DIR

logger = logging.getLogger(__name__)


def compute_ev_coverage_metrics(
    state: str,
    pcs_2024: Optional[int],
    omc_2025: Optional[int],
    ev_count_2023: Optional[int],
    total_vehicles_2023: Optional[int],
) -> Dict[str, Any]:
    """
    Calculate state-level EV penetration and charger availability rates.
    Handles missing datasets (e.g. Telangana in e-Vahan 2023) cleanly.
    """
    pcs_val = int(pcs_2024) if pcs_2024 is not None else 0
    omc_val = int(omc_2025) if omc_2025 is not None else 0

    if ev_count_2023 is None or total_vehicles_2023 is None or ev_count_2023 == 0:
        return {
            "state": state,
            "pcs_2024": pcs_val,
            "omc_2025": omc_val,
            "ev_count_2023": "Not available: dataset missing",
            "total_vehicles_2023": "Not available: dataset missing",
            "ev_share_pct": "Not available: dataset missing",
            "chargers_per_1000_evs_2024": "Not available: dataset missing",
            "chargers_per_1000_evs_2025": "Not available: dataset missing",
        }

    evs = float(ev_count_2023)
    tot = float(total_vehicles_2023)
    ev_share = round((evs / tot) * 100.0, 3) if tot > 0 else 0.0

    chargers_per_1000_pcs = round((pcs_val / evs) * 1000.0, 2)
    chargers_per_1000_omc = round((omc_val / evs) * 1000.0, 2)

    return {
        "state": state,
        "pcs_2024": pcs_val,
        "omc_2025": omc_val,
        "ev_count_2023": int(evs),
        "total_vehicles_2023": int(tot),
        "ev_share_pct": ev_share,
        "chargers_per_1000_evs_2024": chargers_per_1000_pcs,
        "chargers_per_1000_evs_2025": chargers_per_1000_omc,
    }


def get_data_limitations_notice() -> List[Dict[str, str]]:
    """Return explicit notices regarding government dataset boundaries and methodologies."""
    return [
        {
            "category": "Definition Divergence",
            "title": "Different Definitions Warning",
            "description": (
                "The 2024 table measures Operational Public Charging Stations (PCS) under the Ministry of Heavy "
                "Industries, while the 2025 table measures EV charging stations installed by PSU Oil Marketing Companies "
                "(OMCs) at retail fuel outlets. These tables represent differing methodologies and must NEVER be "
                "interpreted as a continuous single time series."
            ),
        },
        {
            "category": "Temporal Alignment",
            "title": "Date Gap Warning",
            "description": (
                "Registered EV figures from the Ministry of Road Transport & Highways reflect data as of 14-07-2023. "
                "Charging infrastructure tables reflect 31-03-2024 and 01-01-2025. Ratios calculated across these periods "
                "are indicative approximations spanning divergent snapshot dates."
            ),
        },
        {
            "category": "Data Completeness",
            "title": "Telangana Missing from Central Portal",
            "description": (
                "Telangana is entirely absent from the official data.gov.in e-Vahan 2023 resource. "
                "Per Rule D3, values are displayed as 'Not available: dataset missing' without synthetic interpolation."
            ),
        },
        {
            "category": "Spatial Granularity",
            "title": "No Station-Level Coordinates",
            "description": (
                "Government open datasets provide charging numbers aggregated strictly at State and Highway Corridor grains. "
                "No station-level GPS coordinates exist on data.gov.in. Consequently, route routing and live availability "
                "cannot be claimed or simulated."
            ),
        },
    ]


def get_ev_state_comparison(conn: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    """Query integrated state mart and return coverage explorer metrics for target states."""
    df = conn.execute("""
        SELECT
            norm_state,
            ev_operational_pcs_2024,
            ev_omc_stations_2025,
            ev_registered_count_2023,
            total_registered_vehicles_2023,
            state_persondays_2024_25,
            state_msme_enterprises
        FROM integrated_states
        ORDER BY norm_state
    """).fetchdf()

    results = []
    for _, row in df.iterrows():
        st = row["norm_state"]
        pcs = row["ev_operational_pcs_2024"]
        omc = row["ev_omc_stations_2025"]
        evs = row["ev_registered_count_2023"] if pd.notna(row["ev_registered_count_2023"]) else None
        tot = row["total_registered_vehicles_2023"] if pd.notna(row["total_registered_vehicles_2023"]) else None

        metrics = compute_ev_coverage_metrics(st, pcs, omc, evs, tot)
        # Include joined exploratory context from MGNREGA & MSME (labeled strictly as exploratory context)
        metrics["exploratory_mgnrega_persondays"] = int(row["state_persondays_2024_25"])
        metrics["exploratory_msme_enterprises"] = int(row["state_msme_enterprises"])
        results.append(metrics)

    return results


def get_highway_corridors(conn: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    """Retrieve the 16 national highway and expressway charging corridors as of 2022."""
    df = conn.execute("""
        SELECT corridor_name, station_count
        FROM clean_ev_corridor_2022
        ORDER BY station_count DESC
    """).fetchdf()

    return df.to_dict(orient="records")
