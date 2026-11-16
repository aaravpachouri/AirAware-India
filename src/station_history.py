"""Append-only, deduplicated OpenAQ station-hour observations."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[1] / "data" / "station_hourly_history.jsonl"


def _utc_iso(value: Any) -> str:
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _hour_key(value: Any) -> str:
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ).isoformat().replace("+00:00", "Z")


def append_station_observations(
    station: dict,
    observations: list[dict],
    path: Path = DEFAULT_HISTORY_PATH,
) -> dict:
    """Persist each station/sensor/pollutant/hour once; skip invalid readings."""
    station_id = int(station["id"])
    path = Path(path)
    existing: dict[tuple[int, int, str, str], dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (
                        int(record["station_id"]),
                        int(record["sensor_id"]),
                        str(record["pollutant"]),
                        _hour_key(record["timestamp"]),
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid station history record at line {line_number}"
                    ) from exc
                existing[key] = record

    seen = 0
    added = 0
    for item in observations:
        seen += 1
        try:
            pollutant = str(item["pollutant"]).strip().lower()
            sensor_id = int(item["sensor_id"])
            timestamp = _utc_iso(item["timestamp"])
            value = float(item["value"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if pollutant not in {"pm25", "pm10", "no2", "so2", "co", "o3"}:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        key = (station_id, sensor_id, pollutant, _hour_key(timestamp))
        if key in existing:
            try:
                existing_stamp = _utc_iso(existing[key]["timestamp"])
            except (KeyError, TypeError, ValueError):
                existing_stamp = ""
            if timestamp <= existing_stamp:
                continue
        existing[key] = {
            "schema_version": 1,
            "source": "OpenAQ v3 hourly sensor measurements",
            "station_id": station_id,
            "station_name": str(station.get("name") or f"Station {station_id}"),
            "city": str(station.get("city") or ""),
            "state": str(station.get("state") or ""),
            "latitude": station.get("latitude"),
            "longitude": station.get("longitude"),
            "sensor_id": sensor_id,
            "pollutant": pollutant,
            "timestamp": timestamp,
            "value": value,
            "unit_basis": "normalized CPCB breakpoint units (µg/m³; CO mg/m³)",
        }
        added += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(existing):
            handle.write(
                json.dumps(existing[key], sort_keys=True, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)
    return {
        "observations_seen": seen,
        "records_added": added,
        "records_total": len(existing),
        "history_path": str(path),
    }
