from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE = "https://airquality.xkdr.org"
OUT = Path("reports/xkdr_source_audit.json")


def request_json(session: requests.Session, path: str, params: dict | None = None):
    response = session.get(f"{BASE}{path}", params=params, timeout=(10, 90))
    response.raise_for_status()
    return response.json(), response.headers


def rows_from(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("stations", "results", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def main():
    key = os.environ.get("XKDR_API_KEY", "").strip()
    if not key:
        raise SystemExit("XKDR_API_KEY is missing; configure it as an Actions secret.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
    # Never log the key or request headers.
    meta, _ = request_json(session, "/v1/meta")
    stations_payload, _ = request_json(
        session, "/v1/stations", {"source": "cpcb_caaqm", "format": "json"}
    )
    stations = rows_from(stations_payload)
    if not stations:
        raise RuntimeError("XKDR station registry returned no parseable station rows.")
    station = stations[0]
    station_id = station.get("station_id") or station.get("id")
    if not station_id:
        raise RuntimeError("XKDR station registry row has no station_id/id.")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=6)
    measurements, headers = request_json(session, "/v1/measurements", {
        "station": station_id,
        "parameter": ["PM2.5", "PM10"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "agg": "hourly",
        "format": "json",
        "limit": 1000,
    })
    sample_rows = rows_from(measurements)
    if not sample_rows:
        raise RuntimeError("XKDR sample measurement query returned no parseable rows.")
    report = {
        "schema_version": 1,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source": "India Air Quality Database, XKDR Forum",
        "source_url": BASE,
        "source_license": "CC BY 4.0; attribution required",
        "upstream_network": "CPCB CAAQM",
        "station_registry_rows": len(stations),
        "station_sample": {
            "station_id": station_id,
            "station_name": station.get("station_name") or station.get("name"),
            "state": station.get("state_name") or station.get("state"),
            "city": station.get("city_name") or station.get("city"),
            "parameters": station.get("parameters"),
        },
        "meta": meta,
        "sample_query": {
            "start_ist_date": start.isoformat(),
            "end_ist_date": end.isoformat(),
            "parameters": ["PM2.5", "PM10"],
            "aggregation": "hourly",
            "rows_returned": len(sample_rows),
            "row_count_header": headers.get("X-Row-Count"),
            "truncated_header": headers.get("X-Truncated"),
        },
        "status": "source_access_verified",
        "limitations": [
            "Upstream readings are published as received and are not independently regulator-validated by XKDR.",
            "This audit does not merge records into AirAware training history.",
            "CPCB station IDs must be reconciled to OpenAQ location IDs before joint station-level training.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "station_registry_rows": len(stations),
        "sample_station_id": station_id,
        "sample_measurement_rows": len(sample_rows),
        "report": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"XKDR source audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
