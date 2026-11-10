"""Prospective same-station validation for AirAware's live AQI forecast."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LEDGER_PATH = ROOT / "data" / "prospective_station_validation.jsonl"
MODEL_PATH = ROOT / "models" / "airaware_live_rf.joblib"
MAX_STATIONS = max(1, min(int(os.getenv("AIR_AWARE_MAX_STATIONS", "25")), 100))
MAX_AGE_HOURS = 6.0
MIN_HORIZON_HOURS = 18.0
MAX_HORIZON_HOURS = 30.0
MIN_POLLUTANTS = 3


def as_utc(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Invalid timestamp.")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").to_pydatetime()


def read_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid ledger JSON at line {number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Ledger line {number} is not an object.")
                records.append(value)
    return records


def write_ledger(records: list[dict], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def eligible_aqi(data: dict, now: datetime) -> tuple[int, list[str]] | None:
    """Require a fresh snapshot and at least 3 subindices including particulate matter."""
    age = data.get("age_hours")
    if age is None or not 0 <= float(age) <= MAX_AGE_HOURS:
        return None
    from live_air import estimate_cpcb_style_aqi
    result = estimate_cpcb_style_aqi(data.get("pollutants") or {})
    subindices = result.get("subindices") or {}
    names = sorted(subindices)
    if len(names) < MIN_POLLUTANTS or not {"pm25", "pm10"}.intersection(names):
        return None
    if result.get("aqi") is None:
        return None
    observed_at = as_utc(data["timestamp"])
    if observed_at > now + timedelta(minutes=5):
        return None
    return int(result["aqi"]), names


def forecast_id(station_id: int, issued_at: datetime) -> str:
    return f"openaq:{int(station_id)}:{issued_at.date().isoformat()}"


def pair_observations(records: list[dict], observations: dict[int, dict], now: datetime) -> int:
    paired = 0
    for record in records:
        if record.get("status") != "pending":
            continue
        station_id = int(record["station"]["id"])
        observation = observations.get(station_id)
        if observation is None:
            continue
        observed_at = as_utc(observation["timestamp"])
        issued_at = as_utc(record["issued_at"])
        horizon = (observed_at - issued_at).total_seconds() / 3600
        if not MIN_HORIZON_HOURS <= horizon <= MAX_HORIZON_HOURS:
            continue
        if observed_at > now + timedelta(minutes=5):
            continue
        actual = int(observation["aqi"])
        record.update({
            "status": "evaluated",
            "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
            "observed_aqi": actual,
            "observed_pollutants": sorted(observation["pollutants"]),
            "actual_horizon_hours": round(horizon, 2),
            "absolute_error": round(abs(float(record["predicted_aqi"]) - actual), 3),
            "persistence_absolute_error": round(abs(float(record["persistence_aqi"]) - actual), 3),
            "evaluation_method": "Same OpenAQ location; later fresh snapshot; >=3 available subindices including PM2.5 or PM10.",
        })
        paired += 1
    return paired


def make_record(station: dict, data: dict, prediction: float, current_aqi: int,
                pollutants: list[str], issued_at: datetime, model_hash: str) -> dict:
    station_id = int(data.get("location_id", station["id"]))
    return {
        "schema_version": 1,
        "forecast_id": forecast_id(station_id, issued_at),
        "status": "pending",
        "source": "OpenAQ v3",
        "source_url": f"https://api.openaq.org/v3/locations/{station_id}",
        "model": "airaware_live_rf.joblib",
        "model_sha256": model_hash,
        "station": {
            "id": station_id,
            "name": data.get("location") or station.get("name") or f"Station {station_id}",
            "city": data.get("city") or station.get("city") or "",
            "state": data.get("state") or station.get("state") or "",
            "country": data.get("country") or station.get("country") or "India",
            "provider": data.get("provider") or station.get("provider") or "Unknown",
            "latitude": data.get("latitude", station.get("latitude")),
            "longitude": data.get("longitude", station.get("longitude")),
        },
        "issued_at": issued_at.isoformat().replace("+00:00", "Z"),
        "target_at": (issued_at + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
        "station_measurement_at_issue": as_utc(data["timestamp"]).isoformat().replace("+00:00", "Z"),
        "station_data_age_hours": round(float(data["age_hours"]), 2),
        "current_aqi": int(current_aqi),
        "persistence_aqi": int(current_aqi),
        "predicted_aqi": round(max(0.0, min(500.0, float(prediction))), 2),
        "available_pollutant_subindices": sorted(pollutants),
        "pollutant_count": len(pollutants),
        "aqi_basis": "Maximum available CPCB-style sub-index; >=3 pollutants including PM2.5 or PM10.",
        "forecast_horizon_hours": 24,
        "validation_note": "Experimental OpenAQ station-snapshot validation, not official CPCB AQI or a regulatory 24-hour average.",
    }


def run_collection(now: datetime | None = None, ledger_path: Path = LEDGER_PATH) -> dict:
    from live_air import get_available_locations, get_live_air_quality
    from live_forecast_features import build_live_forecast_features

    current_time = as_utc(now or datetime.now(timezone.utc))
    if not os.getenv("OPENAQ_API_KEY"):
        raise RuntimeError("OPENAQ_API_KEY is missing; add it as a GitHub Actions repository secret.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing live model: {MODEL_PATH}")
    model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    model = joblib.load(MODEL_PATH)
    records = read_ledger(ledger_path)
    stations = get_available_locations(max_pages=5, max_age_days=2)
    stations = sorted(stations, key=lambda x: x.get("last_datetime") or "", reverse=True)[:MAX_STATIONS]

    snapshots: dict[int, dict] = {}
    data_by_id: dict[int, dict] = {}
    errors = []
    for station in stations:
        station_id = int(station["id"])
        try:
            data = get_live_air_quality(station_id, station)
            valid = eligible_aqi(data, current_time)
            if valid is not None:
                aqi, pollutants = valid
                snapshots[station_id] = {"timestamp": data["timestamp"], "aqi": aqi, "pollutants": pollutants}
                data_by_id[station_id] = data
        except Exception as exc:
            errors.append({"station_id": station_id, "error": str(exc)[:200]})

    paired = pair_observations(records, snapshots, current_time)
    seen = {item.get("forecast_id") for item in records}
    created = 0
    for station in stations:
        station_id = int(station["id"])
        data = data_by_id.get(station_id)
        snapshot = snapshots.get(station_id)
        if data is None or snapshot is None or forecast_id(station_id, current_time) in seen:
            continue
        try:
            features = build_live_forecast_features(
                current_aqi=snapshot["aqi"],
                pollutants=data.get("pollutants") or {},
                timestamp=data["timestamp"],
            )
            prediction = float(model.predict(features)[0])
            records.append(make_record(station, data, prediction, snapshot["aqi"],
                                       snapshot["pollutants"], current_time, model_hash))
            created += 1
        except Exception as exc:
            errors.append({"station_id": station_id, "error": str(exc)[:200]})

    write_ledger(records, ledger_path)
    return {
        "stations_considered": len(stations),
        "eligible_station_snapshots": len(snapshots),
        "forecasts_created": created,
        "forecasts_paired": paired,
        "evaluated_forecasts_total": sum(x.get("status") == "evaluated" for x in records),
        "pending_forecasts_total": sum(x.get("status") == "pending" for x in records),
        "station_errors": errors,
        "ledger": str(ledger_path),
    }


if __name__ == "__main__":
    print(json.dumps(run_collection(), indent=2))
