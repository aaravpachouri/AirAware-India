from __future__ import annotations

"""Download XKDR CPCB monthly Parquet partitions into an isolated, auditable dataset.

This deliberately does not merge CPCB station IDs into OpenAQ history or train a
model. Raw values and source units are preserved; eligibility is a separate QA flag.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import duckdb
import requests

BASE = "https://airquality.xkdr.org"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "xkdr_cpcb"
DEFAULT_CACHE = ROOT / ".cache" / "xkdr"
PARAMETERS = {
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone",
    "Benzene", "Toluene", "Xylene", "O_Xylene", "MP-Xylene", "Eth-Benzene",
}
# XKDR says units are harmonised in spelling only. These are labels, not
# conversions: concentrations are never silently converted between units.
KNOWN_UNITS = {"µg/m³", "ug/m3", "μg/m³", "mg/m³", "mg/m3", "ppb", "ppm"}
IDENTIFIER = re.compile(r"^v1/measurements/year=(20\\d{2})/month=(0[1-9]|1[0-2])/data\\.parquet$")


def parse_month(value: str) -> tuple[int, int]:
    try:
        year, month = map(int, value.split("-", 1))
        date(year, month, 1)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected YYYY-MM") from exc
    return year, month


def month_range(start: tuple[int, int], end: tuple[int, int]):
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def api_session(key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
    return session


def get_json(session: requests.Session, path: str):
    response = session.get(f"{BASE}{path}", timeout=(15, 90))
    response.raise_for_status()
    return response.json()


def file_manifest(session: requests.Session) -> dict:
    payload = get_json(session, "/v1/files")
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise RuntimeError("XKDR /v1/files response has no files array")
    indexed = {}
    for item in files:
        key = item.get("key") if isinstance(item, dict) else None
        match = IDENTIFIER.match(str(key or ""))
        if match:
            indexed[(int(match.group(1)), int(match.group(2)))] = item
    if not indexed:
        raise RuntimeError("No monthly measurement Parquet partitions were advertised")
    return {"payload": payload, "partitions": indexed}


def download_partition(session, key: str, destination: Path, retries: int = 4):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    url = f"{BASE}/v1/files/{quote(key, safe='/=')}"
    last_error = None
    for attempt in range(retries):
        temp = destination.with_suffix(destination.suffix + ".part")
        try:
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                with temp.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
            if temp.stat().st_size < 100:
                raise RuntimeError(f"Unexpectedly small Parquet partition: {key}")
            temp.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {key}: {last_error}") from last_error


def station_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for field in ("stations", "results", "data", "items"):
            if isinstance(payload.get(field), list):
                return payload[field]
    raise RuntimeError("XKDR /v1/stations response has no station rows")


def ingest(start, end, out_dir: Path, cache_dir: Path, key: str):
    session = api_session(key)
    metadata_payload = get_json(session, "/v1/stations?source=cpcb_caaqm&format=json")
    stations = station_rows(metadata_payload)
    # Preserve provider IDs verbatim. Coordinates and names are metadata, not a
    # guessed crosswalk to OpenAQ IDs.
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "cpcb_station_registry.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    advertised = file_manifest(session)
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    processed, skipped = [], []
    for year, month in month_range(start, end):
        item = advertised["partitions"].get((year, month))
        if item is None:
            skipped.append(f"{year:04d}-{month:02d}: not present in provider file manifest")
            continue
        key_path = item["key"]
        local = cache_dir / f"{year:04d}" / f"{month:02d}.parquet"
        download_partition(session, key_path, local)
        target = out_dir / f"cpcb_measurements_{year:04d}_{month:02d}.parquet"
        escaped = str(local.resolve()).replace("'", "''")
        target_sql = str(target.resolve()).replace("'", "''")
        # collected_at is a naive IST timestamp according to XKDR's docs.
        # Retain it as IST wall time and derive an explicit UTC instant.
        query = f"""
          COPY (
            WITH source_rows AS (
              SELECT
                CAST(station_id AS VARCHAR) AS station_id,
                CAST(station_name AS VARCHAR) AS station_name,
                CAST(state_name AS VARCHAR) AS state_name,
                CAST(city_name AS VARCHAR) AS city_name,
                CAST(parameter_name AS VARCHAR) AS parameter_name,
                CAST(unit AS VARCHAR) AS source_unit,
                CAST(collected_at AS TIMESTAMP) AS collected_at_ist,
                (CAST(collected_at AS TIMESTAMP) AT TIME ZONE 'Asia/Kolkata') AS collected_at_utc,
                TRY_CAST(value AS DOUBLE) AS value,
                CAST(source AS VARCHAR) AS source
              FROM read_parquet('{escaped}')
              WHERE source = 'cpcb_caaqm'
            ), classified AS (
              SELECT *,
                CASE
                  WHEN parameter_name NOT IN ({",".join("'" + p.replace("'", "''") + "'" for p in sorted(PARAMETERS))}) THEN 'unsupported_parameter'
                  WHEN value IS NULL OR NOT isfinite(value) THEN 'invalid_numeric'
                  WHEN value < 0 THEN 'negative_value'
                  WHEN source_unit NOT IN ({",".join("'" + u.replace("'", "''") + "'" for u in sorted(KNOWN_UNITS))}) THEN 'unknown_unit'
                  ELSE 'eligible_raw_observation'
                END AS quality_status
              FROM source_rows
            )
            SELECT * EXCLUDE (duplicate_rank) FROM (
              SELECT *, row_number() OVER (
                PARTITION BY station_id, parameter_name, collected_at_ist, source_unit, value
                ORDER BY station_name, city_name
              ) AS duplicate_rank
              FROM classified
            ) WHERE duplicate_rank = 1
          ) TO '{target_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        conn.execute(query)
        counts = conn.execute(f"""
          SELECT count(*) AS rows,
                 count(*) FILTER (WHERE quality_status='eligible_raw_observation') AS eligible_rows,
                 count(DISTINCT station_id) AS stations,
                 count(DISTINCT parameter_name) AS parameters
          FROM read_parquet('{target_sql}')
        """).fetchone()
        processed.append({
            "month": f"{year:04d}-{month:02d}", "file": target.name,
            "source_partition": key_path, "source_bytes": local.stat().st_size,
            "rows": counts[0], "eligible_raw_rows": counts[1],
            "stations": counts[2], "parameters": counts[3],
        })
    conn.close()
    if not processed:
        raise RuntimeError("No XKDR partitions were ingested for the requested range")
    manifest = {
        "schema_version": 1,
        "dataset": "XKDR India Air Quality Database - CPCB CAAQM",
        "source_url": BASE,
        "source_license": "CC BY 4.0",
        "required_attribution": "XKDR Forum, India Air Quality Database; Central Pollution Control Board (CPCB) CAAQM network",
        "source_documentation": "https://airquality.xkdr.org/#docs",
        "source_filter": "source = cpcb_caaqm",
        "requested_start_month": f"{start[0]:04d}-{start[1]:02d}",
        "requested_end_month": f"{end[0]:04d}-{end[1]:02d}",
        "ingested_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "station_registry_rows": len(stations),
        "station_registry_file": metadata_path.name,
        "timestamp_policy": "collected_at is naive Asia/Kolkata wall time; collected_at_utc is derived using Asia/Kolkata timezone",
        "value_policy": "source values and units preserved; no unit conversion, imputation, gap filling, calibration, or outlier removal",
        "deduplication": "exact duplicate station/parameter/timestamp/unit/value rows only",
        "quality_statuses": ["eligible_raw_observation", "unsupported_parameter", "invalid_numeric", "negative_value", "unknown_unit"],
        "station_identity_policy": "XKDR station_id remains separate; no inferred OpenAQ station crosswalk",
        "partitions": processed,
        "skipped_months": skipped,
        "total_rows": sum(p["rows"] for p in processed),
        "total_eligible_raw_rows": sum(p["eligible_raw_rows"] for p in processed),
        "total_stations_observed": len(set()),
        "limitations": [
            "XKDR republishes readings as received; readings are not independently validated by XKDR.",
            "Eligibility is a raw-data QA flag, not confirmation of regulatory validity.",
            "This ingestion does not train or alter production forecast models.",
        ],
    }
    # Calculate the actual union of stations across output partitions.
    conn = duckdb.connect()
    paths = [str(out_dir / p["file"]).replace("'", "''") for p in processed]
    union = ",".join("'" + p + "'" for p in paths)
    manifest["total_stations_observed"] = conn.execute(
        f"SELECT count(DISTINCT station_id) FROM read_parquet([{union}])"
    ).fetchone()[0]
    conn.close()
    (out_dir / "ingestion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_month, default=parse_month("2009-01"))
    parser.add_argument("--end", type=parse_month, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    end = args.end or (date.today().year, date.today().month)
    if args.start > end:
        parser.error("--start must not be after --end")
    key = os.environ.get("XKDR_API_KEY", "").strip()
    if not key:
        parser.error("Set XKDR_API_KEY in the environment; never put it on the command line.")
    report = ingest(args.start, end, args.output, args.cache, key)
    print(json.dumps({
        "status": "ingestion_complete",
        "partitions": len(report["partitions"]),
        "rows": report["total_rows"],
        "eligible_raw_rows": report["total_eligible_raw_rows"],
        "stations": report["total_stations_observed"],
        "output": str(args.output),
        "manifest": str(args.output / "ingestion_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"XKDR ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
