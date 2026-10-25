"""
Data Retrieval Module for South India Open Data Integration Engine.
Strictly fetches from official data.gov.in API or registers verified local CSVs.
Adheres to Absolute Data Rules D1-D10, secret hygiene, backoff and caching.
"""

import os
import sys
import time
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

from config import (
    OGD_API_KEY,
    RAW_DIR,
    WORK_DIR,
    DATASETS,
    TARGET_STATES,
    USER_AGENT,
    sanitize_url,
)

# Configure logging with secret hygiene (never print query string / keys)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fetch_data")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for integrity and provenance tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_cached(filename: str, raw_dir: Path = RAW_DIR) -> bool:
    """Check if the raw dataset file is already downloaded and non-empty."""
    path = raw_dir / filename
    return path.exists() and path.stat().st_size > 0


def update_manifest(
    dataset_key: str,
    resource_id: str,
    title: str,
    url: str,
    filename: str,
    row_count: int,
    sha256_hash: str,
    manifest_path: Path = RAW_DIR / "manifest.json",
) -> Dict[str, Any]:
    """
    Update data/raw/real/manifest.json with verified dataset metadata.
    Logs provenance, exact row count, and cryptographic hash.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    entry = {
        "id": resource_id,
        "title": title,
        "url": sanitize_url(url),
        "filename": filename,
        "row_count": row_count,
        "sha256": sha256_hash,
        "fetch_date": datetime.now(timezone.utc).isoformat(),
    }
    manifest[dataset_key] = entry

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Updated manifest for %s (rows: %d, sha256: %s)", dataset_key, row_count, sha256_hash[:12])
    return entry


def fetch_with_retry(
    url: str,
    params: Dict[str, Any],
    max_retries: int = 4,
    initial_delay: float = 1.0,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """
    Perform an HTTP GET request with exponential backoff and polite delay.
    Guarantees secrets in query parameters are never logged (Rule D4, D5).
    """
    headers = {"User-Agent": USER_AGENT}
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    "HTTP %d on attempt %d/%d for %s. Backing off for %.1fs...",
                    resp.status_code,
                    attempt,
                    max_retries,
                    sanitize_url(url),
                    delay,
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                logger.error(
                    "HTTP %d error for %s: %s",
                    resp.status_code,
                    sanitize_url(url),
                    resp.text[:200],
                )
                return None
        except requests.RequestException as e:
            logger.warning(
                "Request error on attempt %d/%d for %s: %s. Retrying in %.1fs...",
                attempt,
                max_retries,
                sanitize_url(url),
                type(e).__name__,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0

    logger.error("Failed to fetch %s after %d retries.", sanitize_url(url), max_retries)
    return None


def fetch_resource_paginated(
    resource_id: str,
    api_key: str,
    extra_filters: Optional[Dict[str, str]] = None,
    batch_size: int = 1000,
    max_records: Optional[int] = None,
    delay_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Paginate through data.gov.in resource API using limit and offset.
    Respects rate limits, collects all matching records, returns list of records.
    """
    base_url = f"https://api.data.gov.in/resource/{resource_id}"
    offset = 0
    records: List[Dict[str, Any]] = []

    logger.info("Starting paginated retrieval for %s (batch_size: %d)", sanitize_url(base_url), batch_size)

    while True:
        params: Dict[str, Any] = {
            "api-key": api_key,
            "format": "json",
            "limit": batch_size,
            "offset": offset,
        }
        if extra_filters:
            for k, v in extra_filters.items():
                params[f"filters[{k}]"] = v

        resp = fetch_with_retry(base_url, params=params)
        if not resp:
            logger.error("Pagination aborted due to network failure at offset %d", offset)
            break

        try:
            data = resp.json()
        except Exception as e:
            logger.error("JSON decode error at offset %d: %s", offset, e)
            break

        batch = data.get("records", [])
        total_reported = data.get("total", None)

        if not batch:
            logger.info("No further records returned at offset %d. Finished pagination.", offset)
            break

        records.extend(batch)
        logger.info(
            "Retrieved %d records (total so far: %d / %s)",
            len(batch),
            len(records),
            str(total_reported) if total_reported is not None else "unknown",
        )

        offset += len(batch)

        if max_records and len(records) >= max_records:
            records = records[:max_records]
            logger.info("Reached maximum record cap (%d).", max_records)
            break

        if total_reported is not None and offset >= int(total_reported):
            break

        time.sleep(delay_sec)

    return records


def register_local_csv(
    dataset_key: str,
    resource_id: str,
    title: str,
    url: str,
    csv_path: Path,
    manifest_path: Path = RAW_DIR / "manifest.json",
) -> Dict[str, Any]:
    """
    Register a manually placed CSV dataset into manifest.json.
    Verifies file existence, counts records, and logs SHA-256 hash.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Local file not found: {csv_path}")

    # Count lines excluding header
    line_count = 0
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 0 and line.strip():
                line_count += 1

    sha = compute_sha256(csv_path)
    entry = update_manifest(
        dataset_key=dataset_key,
        resource_id=resource_id,
        title=title,
        url=url,
        filename=csv_path.name,
        row_count=line_count,
        sha256_hash=sha,
        manifest_path=manifest_path,
    )
    logger.info("Manually registered CSV %s (%d rows)", csv_path.name, line_count)
    return entry


def fetch_all(force: bool = False) -> Dict[str, Any]:
    """
    Fetch all verified datasets in DATASETS manifest.
    Resumable: skips datasets whose cached JSON file already exists unless force=True.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if not OGD_API_KEY:
        logger.error("OGD_API_KEY is missing or empty in .env. Cannot call official API.")
        return {}

    summary = {}

    for key, meta in DATASETS.items():
        local_file = RAW_DIR / meta["local_filename"]

        if is_cached(meta["local_filename"]) and not force:
            logger.info("Skipping [%s]: already cached at %s", key, local_file.name)
            summary[key] = {
                "status": "CACHED",
                "filename": meta["local_filename"],
                "row_count": len(json.loads(local_file.read_text())),
            }
            continue

        if meta.get("access_method") != "api":
            logger.info("Dataset [%s] requires manual download. See MANUAL_DOWNLOADS.", key)
            summary[key] = {"status": "MANUAL DOWNLOAD NEEDED", "url": meta["page_url"]}
            continue

        logger.info("=== Fetching dataset: %s (%s) ===", key, meta["title"])

        # Fetch records
        # For large tables (pincode, mgnrega), filter by target states or fetch per state
        all_records: List[Dict[str, Any]] = []

        if meta.get("state_col") and key in ("pincode_directory", "mgnrega_district"):
            # Fetch per South Indian state to avoid downloading 400k unrelated national rows
            state_col = meta["state_col"]
            for state in TARGET_STATES:
                logger.info("Fetching %s for target state: %s", key, state)
                state_records = fetch_resource_paginated(
                    resource_id=meta["id"],
                    api_key=OGD_API_KEY,
                    extra_filters={state_col: state},
                    batch_size=1000,
                    delay_sec=0.4,
                )
                all_records.extend(state_records)
                time.sleep(0.3)
        else:
            # Pan-India table (EV tables are 16-37 rows total; MSME is 707 rows total)
            all_records = fetch_resource_paginated(
                resource_id=meta["id"],
                api_key=OGD_API_KEY,
                batch_size=1000,
                delay_sec=0.3,
            )

        if not all_records:
            logger.warning("No records fetched for [%s]", key)
            summary[key] = {"status": "EMPTY OR FAILED"}
            continue

        # Save cached JSON to data/raw/real/
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)

        sha = compute_sha256(local_file)
        update_manifest(
            dataset_key=key,
            resource_id=meta["id"],
            title=meta["title"],
            url=meta["page_url"],
            filename=local_file.name,
            row_count=len(all_records),
            sha256_hash=sha,
        )

        summary[key] = {
            "status": "FETCHED",
            "filename": local_file.name,
            "row_count": len(all_records),
            "sha256": sha,
        }
        logger.info("Successfully saved %s (%d records) -> %s", key, len(all_records), local_file.name)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and register data.gov.in datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--register", action="store_true", help="Register manual CSVs in data/raw/real/")
    args = parser.parse_args()

    if args.register:
        logger.info("Scanning data/raw/real/ for manual CSV registration...")
        for csv_file in RAW_DIR.glob("*.csv"):
            logger.info("Found local CSV: %s", csv_file.name)
    else:
        fetch_all(force=args.force)
