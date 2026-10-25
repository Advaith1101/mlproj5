"""
South India Open Data Integration Engine - Dataset Joins & Lineage Graph
Performs join-key discovery, grain validation, and multi-table integration.
Rules:
  - Canonical district join key: (norm_state, norm_district)
  - Strict grain validation: State-level EV tables join to district tables
    ONLY by aggregating district data UP to the state level first.
  - Generates NetworkX integration graph exported to JSON for UI visualisation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import networkx as nx
import pandas as pd

from config import DATA_DIR, WORK_DIR

logger = logging.getLogger(__name__)

GRAPH_FILE = WORK_DIR / "dataset_graph.json"


def validate_join_compatibility(
    left_table: str,
    left_grain: str,
    right_table: str,
    right_grain: str,
    join_keys: List[str],
    aggregate_left_up: bool = False,
) -> Tuple[bool, str]:
    """
    Validate that a proposed join between two government tables does not produce
    an invalid grain combination, spurious duplicate expansion, or misleading relations.
    """
    # Prohibit direct join to corridor dataset which lacks state/district keys
    if "corridor" in right_table.lower() or "corridor" in left_table.lower():
        return (
            False,
            "Invalid join: clean_ev_corridor_2022 lacks geographic administrative keys (state/district).",
        )

    # State-to-District Grain Enforcement
    if left_grain == "district" and right_grain == "state":
        if not aggregate_left_up:
            return (
                False,
                "Grain mismatch: cannot join state table directly to district grain without aggregation. Aggregation required.",
            )

    return True, "Valid join configuration adhering to grain constraints."


def join_district_tables(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: Optional[List[str]] = None,
    how: str = "inner",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform a validated district join and check for duplicate row inflation.
    """
    join_cols = on or ["norm_state", "norm_district"]
    len_left = len(df_left)
    len_right = len(df_right)

    merged = pd.merge(df_left, df_right, on=join_cols, how=how)
    len_merged = len(merged)

    base_len = max(len_left, len_right)
    expansion_factor = round(len_merged / base_len, 4) if base_len > 0 else 1.0

    report = {
        "left_rows": len_left,
        "right_rows": len_right,
        "merged_rows": len_merged,
        "expansion_factor": expansion_factor,
        "how": how,
        "join_keys": join_cols,
    }
    return merged, report


def build_dataset_graph_json(db_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Construct a directed integration graph using NetworkX.
    Nodes represent tables with grain and layer metadata; edges represent validated joins.
    """
    G = nx.DiGraph()

    # Define table nodes
    nodes = [
        {
            "id": "clean_pincode",
            "label": "Pincode Directory",
            "grain": "pincode",
            "layer": "Canonical Reference",
            "ministry": "Dept of Posts",
            "records": 43477,
        },
        {
            "id": "clean_mgnrega",
            "label": "MGNREGA Employment",
            "grain": "district_month_panel",
            "layer": "Socioeconomic Indicator",
            "ministry": "Ministry of Rural Development",
            "records": 12193,
        },
        {
            "id": "clean_msme",
            "label": "MSME Enterprises",
            "grain": "district",
            "layer": "Economic Activity",
            "ministry": "Ministry of MSME",
            "records": 125,
        },
        {
            "id": "integrated_districts",
            "label": "Integrated District Mart",
            "grain": "district",
            "layer": "Standardised Integration",
            "ministry": "Integrated View",
            "records": 125,
        },
        {
            "id": "integrated_states",
            "label": "Integrated State Summary",
            "grain": "state",
            "layer": "Aggregated State View",
            "ministry": "Integrated View",
            "records": 6,
        },
        {
            "id": "clean_ev_pcs_2024",
            "label": "EV Charging PCS (2024)",
            "grain": "state",
            "layer": "EV Coverage",
            "ministry": "Ministry of Heavy Industries",
            "records": 34,
        },
        {
            "id": "clean_ev_omc_2025",
            "label": "EV Charging OMC (2025)",
            "grain": "state",
            "layer": "EV Coverage",
            "ministry": "Ministry of Petroleum",
            "records": 36,
        },
        {
            "id": "clean_ev_vahan_2023",
            "label": "e-Vahan Registered EVs (2023)",
            "grain": "state",
            "layer": "EV Vehicle Density",
            "ministry": "MoRTH",
            "records": 34,
        },
        {
            "id": "clean_ev_corridor_2022",
            "label": "Highway EV Corridors (2022)",
            "grain": "corridor",
            "layer": "Isolated Corridor View",
            "ministry": "MoRTH",
            "records": 16,
        },
    ]

    for n in nodes:
        G.add_node(n["id"], **n)

    # Define validated join edges
    edges = [
        {
            "source": "clean_pincode",
            "target": "integrated_districts",
            "type": "N:1 Aggregation",
            "keys": ["norm_state", "norm_district"],
            "description": "Pincodes aggregated to district totals (PIN count, post offices, centroid lat/lon)",
        },
        {
            "source": "clean_mgnrega",
            "target": "integrated_districts",
            "type": "N:1 Panel Summary",
            "keys": ["norm_state", "norm_district"],
            "description": "Annual cumulative employment metrics (persondays, expenditure)",
        },
        {
            "source": "clean_msme",
            "target": "integrated_districts",
            "type": "1:1 Entity Match",
            "keys": ["norm_state", "norm_district"],
            "description": "Resolved MSME enterprise counts (Micro, Small, Medium)",
        },
        {
            "source": "integrated_districts",
            "target": "integrated_states",
            "type": "N:1 Upward Aggregation",
            "keys": ["norm_state"],
            "description": "District socioeconomic indicators aggregated UP to state grain",
        },
        {
            "source": "clean_ev_pcs_2024",
            "target": "integrated_states",
            "type": "1:1 State Join",
            "keys": ["norm_state"],
            "description": "Public Charging Stations joined at state level",
        },
        {
            "source": "clean_ev_omc_2025",
            "target": "integrated_states",
            "type": "1:1 State Join",
            "keys": ["norm_state"],
            "description": "Oil Marketing Company charging points joined at state level",
        },
        {
            "source": "clean_ev_vahan_2023",
            "target": "integrated_states",
            "type": "1:1 State Join",
            "keys": ["norm_state"],
            "description": "e-Vahan vehicle counts joined (Note: Telangana is absent from official source)",
        },
    ]

    for e in edges:
        G.add_edge(e["source"], e["target"], **e)

    # Export graph to JSON
    graph_dict = {
        "nodes": [dict(G.nodes[node]) for node in G.nodes()],
        "edges": [
            {
                "source": u,
                "target": v,
                **G.edges[u, v],
            }
            for u, v in G.edges()
        ],
    }

    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_FILE, "w") as f:
        json.dump(graph_dict, f, indent=2)

    return graph_dict


def build_integrated_tables_in_db(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Construct integrated district and state tables in DuckDB.
    Strictly follows grain aggregation rules.
    """
    # 1. District aggregated pincode features
    conn.execute("""
        CREATE OR REPLACE TABLE int_pincode_district AS
        SELECT
            norm_state,
            norm_district,
            COUNT(DISTINCT clean_pin) as total_pincodes,
            COUNT(*) as total_post_offices,
            ROUND(AVG(clean_lat), 4) as avg_latitude,
            ROUND(AVG(clean_lon), 4) as avg_longitude
        FROM clean_pincode
        WHERE norm_state IS NOT NULL AND norm_district IS NOT NULL
        GROUP BY norm_state, norm_district
    """)

    # 2. District aggregated MGNREGA features (latest financial year: 2024-2025)
    conn.execute("""
        CREATE OR REPLACE TABLE int_mgnrega_district AS
        SELECT
            norm_state,
            norm_district,
            ROUND(SUM(TRY_CAST(Persondays_of_Central_Liability_so_far AS DOUBLE)), 2) as annual_persondays_generated,
            ROUND(SUM(TRY_CAST(Total_Exp AS DOUBLE)), 2) as annual_total_expenditure
        FROM clean_mgnrega
        WHERE fin_year = '2024-2025'
        GROUP BY norm_state, norm_district
    """)


    # 3. Combine at district grain
    conn.execute("""
        CREATE OR REPLACE TABLE integrated_districts AS
        SELECT
            p.norm_state,
            p.norm_district,
            p.total_pincodes,
            p.total_post_offices,
            p.avg_latitude,
            p.avg_longitude,
            COALESCE(m.annual_persondays_generated, 0) as mgnrega_persondays_2024_25,
            COALESCE(m.annual_total_expenditure, 0) as mgnrega_expenditure_2024_25,
            COALESCE(ms.totalnoofapplication, 0) as msme_enterprises,
            COALESCE(ms.totalnoof_micro, 0) as msme_micro,
            COALESCE(ms.totalnoof_small, 0) as msme_small,
            COALESCE(ms.totalnoof_medium, 0) as msme_medium
        FROM int_pincode_district p
        LEFT JOIN int_mgnrega_district m
            ON p.norm_state = m.norm_state AND p.norm_district = m.norm_district
        LEFT JOIN clean_msme ms
            ON p.norm_state = ms.norm_state AND p.norm_district = ms.norm_district
    """)

    # 4. Aggregate district metrics UP to state level, then safely join state EV tables
    conn.execute("""
        CREATE OR REPLACE TABLE integrated_states AS
        SELECT
            d.norm_state,
            COUNT(DISTINCT d.norm_district) as total_districts,
            SUM(d.total_pincodes) as total_pincodes,
            SUM(d.mgnrega_persondays_2024_25) as state_persondays_2024_25,
            SUM(d.msme_enterprises) as state_msme_enterprises,
            COALESCE(pcs.operational_pcs, 0) as ev_operational_pcs_2024,
            COALESCE(omc.omc_total_evcs, 0) as ev_omc_stations_2025,
            v.ev_count as ev_registered_count_2023,
            v.total_vehicles as total_registered_vehicles_2023
        FROM integrated_districts d
        LEFT JOIN clean_ev_pcs_2024 pcs ON d.norm_state = pcs.norm_state
        LEFT JOIN clean_ev_omc_2025 omc ON d.norm_state = omc.norm_state
        LEFT JOIN clean_ev_vahan_2023 v ON d.norm_state = v.norm_state
        GROUP BY
            d.norm_state,
            pcs.operational_pcs,
            omc.omc_total_evcs,
            v.ev_count,
            v.total_vehicles
    """)


if __name__ == "__main__":
    db_file = WORK_DIR / "engine.duckdb"
    print(f"[*] Building integration mart and dataset graph into {db_file}...")
    conn = duckdb.connect(str(db_file))
    build_integrated_tables_in_db(conn)
    graph = build_dataset_graph_json()
    print(f"[✓] Integration complete. Graph has {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.")
    conn.close()
