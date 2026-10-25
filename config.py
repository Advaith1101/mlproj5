"""
Configuration module for South India Open Data Integration Engine.
Strictly adheres to official data.gov.in sources and secret hygiene.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "real"
WORK_DIR = DATA_DIR / "work"

# Global random seed for reproducible ML operations
SEED = 42

# Target South Indian States and Union Territories (canonical uppercase)
TARGET_STATES = [
    "TAMIL NADU",
    "KARNATAKA",
    "KERALA",
    "ANDHRA PRADESH",
    "TELANGANA",
    "PUDUCHERRY",
]

# State alias map for harmonising state naming variations across portals
STATE_ALIASES = {
    "TAMIL NADU": "TAMIL NADU",
    "TAMILNADU": "TAMIL NADU",
    "KARNATAKA": "KARNATAKA",
    "KERALA": "KERALA",
    "ANDHRA PRADESH": "ANDHRA PRADESH",
    "TELANGANA": "TELANGANA",
    "PUDUCHERRY": "PUDUCHERRY",
    "PONDICHERRY": "PUDUCHERRY",
}

# Secret API key from environment (NEVER print or log this value)
OGD_API_KEY = os.getenv("OGD_API_KEY", "").strip()

# User-Agent header for polite API crawling
USER_AGENT = "SouthIndiaOpenDataEngine/1.0 (academic research; TCS M.Sc. Project)"

# Manifest of verified data.gov.in datasets
# Each entry is verified against data.gov.in live resources
DATASETS = {
    "pincode_directory": {
        "id": "5c2f62fe-5afa-4119-a499-fec9d604d5bd",
        "title": "All India Pincode Directory till last month",
        "ministry": "Department of Posts, Ministry of Communications",
        "page_url": "https://www.data.gov.in/resource/all-india-pincode-directory-till-last-month",
        "api_endpoint": "https://api.data.gov.in/resource/5c2f62fe-5afa-4119-a499-fec9d604d5bd",
        "access_method": "api",
        "local_filename": "pincode_directory.json",
        "state_col": "statename",
        "district_col": "district",
        "has_coordinates": True,
        "feature": "Geographic & Postal Reference Backbone (Core Feature A, B, C)",
    },
    "mgnrega_district": {
        "id": "ee03643a-ee4c-48c2-ac30-9f2ff26ab722",
        "title": "District-wise MGNREGA Data at a Glance",
        "ministry": "Ministry of Rural Development",
        "page_url": "https://www.data.gov.in/resource/district-wise-mgnrega-data-glance",
        "api_endpoint": "https://api.data.gov.in/resource/ee03643a-ee4c-48c2-ac30-9f2ff26ab722",
        "access_method": "api",
        "local_filename": "mgnrega_district.json",
        "state_col": "state_name",
        "district_col": "district_name",
        "has_coordinates": False,
        "feature": "Socio-economic & Rural Employment Indicators (Core Feature A, B, C)",
    },
    "msme_district": {
        "id": "d76a86b1-6a2a-4ab3-9201-cb9f6bc61fa4",
        "title": "District Wise Total MSME Registered Service Enterprises till last date",
        "ministry": "Ministry of Micro, Small and Medium Enterprises",
        "page_url": "https://www.data.gov.in/resource/district-wise-total-msme-registered-service-enterprises-till-last-date",
        "api_endpoint": "https://api.data.gov.in/resource/d76a86b1-6a2a-4ab3-9201-cb9f6bc61fa4",
        "access_method": "api",
        "local_filename": "msme_district.json",
        "state_col": "statename",
        "district_col": "districtname",
        "has_coordinates": False,
        "feature": "Industrial Enterprise Statistics & 3rd District Matching Table (Core Feature B, C)",
    },
    "ev_charging_state_2024": {
        "id": "203a82ce-9b00-4e3a-b18d-daf85e16684f",
        "title": "State/UT-wise Number of Operational Public Electric Vehicles (EVs) Charging Stations (PCS) as on 31st March 2024",
        "ministry": "Ministry of Heavy Industries",
        "page_url": "https://www.data.gov.in/resource/stateut-wise-number-operational-public-electric-vehicles-evs-charging-stations-pcs-31st",
        "api_endpoint": "https://api.data.gov.in/resource/203a82ce-9b00-4e3a-b18d-daf85e16684f",
        "access_method": "api",
        "local_filename": "ev_charging_state_2024.json",
        "state_col": "state__ut",
        "district_col": None,
        "has_coordinates": False,
        "feature": "EV Operational PCS State Coverage Snapshot 2024 (Section 9 - Option A)",
    },
    "ev_charging_omc_2025": {
        "id": "aec86798-2935-47d2-9cda-7438ec2a1c38",
        "title": "State/UT-wise Number of Electric Vehicles Charging Stations (EVCS) installed/energized by PSU OMC as on 01-01-2025",
        "ministry": "Ministry of Petroleum and Natural Gas / Ministry of Heavy Industries",
        "page_url": "https://www.data.gov.in/resource/stateut-wise-number-electric-vehicles-charging-stations-evcs-installedenergized-psu-omc-01",
        "api_endpoint": "https://api.data.gov.in/resource/aec86798-2935-47d2-9cda-7438ec2a1c38",
        "access_method": "api",
        "local_filename": "ev_charging_omc_2025.json",
        "state_col": "state_ut",
        "district_col": None,
        "has_coordinates": False,
        "feature": "EV Charging Stations PSU OMC Growth Snapshot 2025 (Section 9 - Option A)",
    },
    "ev_registered_vehicles_2023": {
        "id": "482a2222-b08a-4c7b-97c1-90ebe63362f5",
        "title": "State/UT-wise Detailed List of Electric Vehicles and Total Vehicles on Roads as per e-Vahan portal as per Vahan4 as on 14-07-2023",
        "ministry": "Ministry of Road Transport and Highways",
        "page_url": "https://www.data.gov.in/resource/stateut-wise-detailed-list-electric-vehicles-and-total-vehicles-roads-e-vahan-portal",
        "api_endpoint": "https://api.data.gov.in/resource/482a2222-b08a-4c7b-97c1-90ebe63362f5",
        "access_method": "api",
        "local_filename": "ev_registered_vehicles_2023.json",
        "state_col": "state_ut",
        "district_col": None,
        "has_coordinates": False,
        "feature": "EV Adoption & Vehicle Fleet Proportion (Section 9 - Option A)",
    },
    "ev_city_highway_2022": {
        "id": "c8158641-7283-4abb-865c-b04fb0b43933",
        "title": "City/Highway-wise Electric Vehicle (EV) Charging Stations out of which 479 Charging Stations have been Installed as on 07-12-2022",
        "ministry": "Ministry of Heavy Industries",
        "page_url": "https://data.gov.in/resource/cityhighway-wise-electric-vehicle-ev-charging-stations-out-which-479-charging-stations",
        "api_endpoint": "https://api.data.gov.in/resource/c8158641-7283-4abb-865c-b04fb0b43933",
        "access_method": "api",
        "local_filename": "ev_city_highway_2022.json",
        "state_col": None,
        "district_col": None,
        "has_coordinates": False,
        "feature": "EV City & Highway Deployment Granularity (Section 9 - Option A)",
    },
}

# Column mapping stubs for standardisation in downstream pipeline.py
COLUMN_MAPS = {
    "pincode_directory": {
        "circlename": "circle_name",
        "regionname": "region_name",
        "divisionname": "division_name",
        "officename": "office_name",
        "pincode": "pincode",
        "officetype": "office_type",
        "delivery": "delivery_status",
        "district": "district_name",
        "statename": "state_name",
        "latitude": "latitude",
        "longitude": "longitude",
    },
    "mgnrega_district": {
        "fin_year": "fin_year",
        "month": "month",
        "state_name": "state_name",
        "district_name": "district_name",
        "Total_No_of_JobCards_issued": "job_cards_issued",
        "Total_No_of_Workers": "workers_total",
        "Total_No_of_Active_Job_Cards": "active_job_cards",
        "Total_No_of_Active_Workers": "active_workers",
    },
    "msme_district": {
        "statename": "state_name",
        "state_id": "state_code",
        "districtname": "district_name",
        "district_id": "district_code",
        "totalnoofapplication": "msme_total_applications",
        "totalnoof_micro": "msme_micro_units",
        "totalnoof_small": "msme_small_units",
        "totalnoof_medium": "msme_medium_units",
    },
    "ev_charging_state_2024": {
        "state__ut": "state_name",
        "no__of_pcs_as_on_31st_march_2024": "operational_pcs_2024",
    },
    "ev_charging_omc_2025": {
        "state_ut": "state_name",
        "total_no__of_ev_charging_stations_installed_by_omcs_from_their_own_funds_as_on_01_01_2025": "omc_evcs_own_funds_2025",
        "ev_charging_stations_under_fame_ii_subsidy_scheme___no__of_ev_charging_stations_energized_as_on_01_01_2025": "fame2_evcs_energized_2025",
    },
    "ev_registered_vehicles_2023": {
        "state_ut": "state_name",
        "till_date_state_wise___total_number_of_vehicles_registered": "total_vehicles_registered",
        "till_date_state_wise___total_vehicle_registered_as_electric": "electric_vehicles_registered",
        "proportion_of_electric_vehicle": "ev_proportion_pct",
    },
    "ev_city_highway_2022": {
        "category": "category",
        "city_highway": "location_name",
        "charging_stations": "installed_chargers",
    },
}


def sanitize_url(url: str) -> str:
    """
    Strip query parameters (especially api-key) from URL for safe logging.
    Ensures OGD_API_KEY is never leaked in logs, stdout or exceptions (Rule D4).
    """
    if "?" in url:
        return url.split("?")[0]
    return url
