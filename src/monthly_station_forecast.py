"""Leakage-safe direct 1..30 day station AQI forecasts and forecast verification."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from station_forecast import (
    FEATURES, POLLUTANTS, _features_at, _utc, aqi_for_window,
    hourly_station_series, read_history,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "airaware_monthly_station.joblib"
REPORT_PATH = ROOT / "reports" / "monthly_station_validation.json"
LEDGER_PATH = ROOT / "data" / "monthly_station_forecast_ledger.jsonl"
MAX_HORIZON_DAYS = 30
MIN_SAMPLES = 500
MIN_STATIONS = 50


def _pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                      max_features=0.9, random_state=42, n_jobs=-1)),
    ])


def build_horizon_frame(records: list[dict], horizon_days: int) -> pd.DataFrame:
    """Daily issue rows; target is a future 24h AQI window at the requested horizon."""
    if not 1 <= int(horizon_days) <= MAX_HORIZON_DAYS:
        raise ValueError("horizon_days must be between 1 and 30")
    rows = []
    for station_id, series in hourly_station_series(records).items():
        hours = sorted({h for values in series.values() for h in values})
        if not hours:
            continue
        anchor = min(hours).replace(minute=0, second=0, microsecond=0) + timedelta(hours=72)
        last = max(hours)
        # Hourly issue times support any time-of-day inference. Validation below
        # samples one fixed UTC hour per station/day to avoid overlapping targets.
        while anchor + timedelta(days=horizon_days) <= last + timedelta(hours=1):
            features = _features_at(series, anchor)
            target, _ = aqi_for_window(series, anchor + timedelta(days=horizon_days))
            if features is not None and target is not None:
                rows.append({"station_id": station_id, "issued_at": anchor.isoformat(),
                             **features, "target_aqi": float(target)})
            anchor += timedelta(hours=1)
    columns = ["station_id", "issued_at", *FEATURES, "target_aqi"]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["issued_at", "station_id"]
    ).reset_index(drop=True) if rows else pd.DataFrame(columns=columns)


def _metrics(frame, prediction):
    actual = frame.target_aqi.to_numpy(float)
    persistence = frame.aqi_current.to_numpy(float)
    rolling = frame.aqi_3day_mean.to_numpy(float)
    model_mae = float(mean_absolute_error(actual, prediction))
    persistence_mae = float(mean_absolute_error(actual, persistence))
    rolling_mae = float(mean_absolute_error(actual, rolling))
    best = min(persistence_mae, rolling_mae)
    return {
        "samples": int(len(frame)), "stations": int(frame.station_id.nunique()),
        "model_mae": round(model_mae, 3),
        "persistence_mae": round(persistence_mae, 3),
        "three_day_mean_mae": round(rolling_mae, 3),
        "best_baseline_mae": round(best, 3),
        "improvement_over_best_baseline_pct": round((best-model_mae)/best*100, 3) if best else None,
        "absolute_error_p90": round(float(np.quantile(np.abs(actual-prediction), .9)), 3),
    }


def _station_spans(records):
    spans = {}
    for row in records:
        try:
            spans.setdefault(int(row["station_id"]), []).append(_utc(row["timestamp"]))
        except (KeyError, TypeError, ValueError):
            pass
    return {sid: (max(v)-min(v)).total_seconds()/86400 for sid, v in spans.items() if v}


def train_monthly_models(records: list[dict], *, model_path=MODEL_PATH,
                         report_path=REPORT_PATH, max_horizon_days=30) -> dict:
    """Each horizon is independently trained, purged and baseline-gated."""
    model_path, report_path = Path(model_path), Path(report_path)
    spans = _station_spans(records)
    models, horizon_reports = {}, {}
    for horizon in range(1, min(int(max_horizon_days), 30)+1):
        frame = build_horizon_frame(records, horizon)
        mature = {sid for sid, days in spans.items() if days >= horizon+3}
        if not frame.empty:
            frame = frame[frame.station_id.isin(mature)].copy()
        frame["issued_at"] = pd.to_datetime(frame.issued_at, utc=True)
        station_ids = sorted(frame.station_id.astype(int).unique()) if not frame.empty else []
        base = {"horizon_days": horizon, "samples": int(len(frame)),
                "stations": len(station_ids), "minimum_samples": MIN_SAMPLES,
                "minimum_stations": MIN_STATIONS, "minimum_history_days_per_station": horizon+3}
        if len(frame) < MIN_SAMPLES or len(station_ids) < MIN_STATIONS:
            horizon_reports[str(horizon)] = {**base, "status": "insufficient_data",
                                              "forecast_enabled": False}
            continue
        split = frame.issued_at.quantile(.8)
        cutoff = split - timedelta(days=horizon+1)
        temporal_train, temporal_test_all = frame[frame.issued_at < cutoff], frame[frame.issued_at >= split]
        held_ids = set(station_ids[-max(1, int(math.ceil(len(station_ids)*.2))):])
        spatial_train = frame[(~frame.station_id.isin(held_ids)) & (frame.issued_at < cutoff)]
        spatial_test_all = frame[frame.station_id.isin(held_ids) & (frame.issued_at >= split)]
        # One fixed UTC hour per station/day gives non-overlapping target windows.
        temporal_hours = (temporal_test_all.station_id.astype(int) % 4) * 6
        spatial_hours = (spatial_test_all.station_id.astype(int) % 4) * 6
        temporal_test = temporal_test_all[temporal_test_all.issued_at.dt.hour == temporal_hours]
        spatial_test = spatial_test_all[spatial_test_all.issued_at.dt.hour == spatial_hours]
        if min(len(temporal_train), len(temporal_test), len(spatial_train), len(spatial_test)) < 20:
            horizon_reports[str(horizon)] = {**base, "status": "insufficient_validation_data",
                                              "forecast_enabled": False}
            continue
        tm, sm = _pipeline(), _pipeline()
        tm.fit(temporal_train[list(FEATURES)], temporal_train.target_aqi)
        sm.fit(spatial_train[list(FEATURES)], spatial_train.target_aqi)
        mt = _metrics(temporal_test, tm.predict(temporal_test[list(FEATURES)]))
        ms = _metrics(spatial_test, sm.predict(spatial_test[list(FEATURES)]))
        tpass = mt["model_mae"] < mt["best_baseline_mae"]*.95
        spass = ms["model_mae"] < ms["best_baseline_mae"]*.95
        passed = bool(tpass and spass)
        horizon_reports[str(horizon)] = {
            **base, "status": "validated" if passed else "baseline_not_beaten",
            "forecast_enabled": passed, "purge_days": horizon+1,
            "temporal_holdout": mt, "station_holdout": ms,
            "station_holdout_ids": sorted(int(station_id) for station_id in held_ids),
            "temporal_gate_passed": bool(tpass), "station_gate_passed": bool(spass),
        }
        if passed:
            final = _pipeline()
            final.fit(frame[list(FEATURES)], frame.target_aqi)
            models[horizon] = {"model": final, "features": list(FEATURES),
                               "validation_status": "validated", "horizon_days": horizon,
                               "residual_p90": max(mt["absolute_error_p90"], ms["absolute_error_p90"])}
    report = {
        "schema_version": 1, "status": "validated" if models else "insufficient_data",
        "source": "OpenAQ v3 station-hour observations",
        "target": "future 24-hour station AQI window; direct independent horizon",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validated_horizons_days": sorted(models),
        "stations_with_history": len(spans), "horizons": horizon_reports,
        "note": "Only individually validated horizons are published. Research estimate, not official CPCB AQI.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    if models:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": 1, "validation_status": "validated",
                     "features": list(FEATURES), "trained_at": report["trained_at"],
                     "models": models}, model_path)
    elif model_path.exists():
        model_path.unlink()
    return report


def _append_event(path: Path, event: dict, existing_ids: set[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    event_id = event["event_id"]
    if event_id in existing_ids:
        return False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":"))+"\n")
    existing_ids.add(event_id)
    return True


def update_forecast_ledger(records, artifact, *, ledger_path=LEDGER_PATH, now=None):
    """Append issued forecasts and score only targets whose future window matured."""
    series_map = hourly_station_series(records)
    anchor = _utc(now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    added = scored = 0
    path = Path(ledger_path)
    existing_ids: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    existing_ids.add(str(json.loads(line).get("event_id", "")))
                except (json.JSONDecodeError, TypeError):
                    continue
    for station_id, series in series_map.items():
        features = _features_at(series, anchor)
        if features is None:
            continue
        for horizon, info in artifact.get("models", {}).items():
            h = int(horizon)
            try:
                prediction = float(info["model"].predict(pd.DataFrame([features], columns=FEATURES))[0])
            except Exception:
                continue
            if not math.isfinite(prediction):
                continue
            residual = max(0.0, float(info.get("residual_p90", 0)))
            event = {
                "event_id": f"forecast:{station_id}:{anchor.isoformat()}:{h}",
                "event_type": "forecast", "station_id": int(station_id),
                "issued_at": anchor.isoformat(), "input_data_cutoff": anchor.isoformat(),
                "horizon_days": h, "target_at": (anchor+timedelta(days=h)).isoformat(),
                "predicted_aqi": round(max(0, min(500, prediction)), 2),
                "lower_90": round(max(0, prediction-residual), 2),
                "upper_90": round(min(500, prediction+residual), 2),
                "model_version": artifact.get("trained_at"), "source": "OpenAQ v3",
            }
            added += int(_append_event(path, event, existing_ids))
    if not path.exists():
        return {"forecast_events_added": added, "verification_events_added": 0}
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    verified = {e.get("forecast_event_id") for e in events if e.get("event_type") == "verification"}
    for forecast in (e for e in events if e.get("event_type") == "forecast"):
        fid = forecast["event_id"]
        if fid in verified:
            continue
        target_at = _utc(forecast["target_at"])
        if anchor < target_at:
            continue
        series = series_map.get(int(forecast["station_id"]))
        actual = aqi_for_window(series, target_at)[0] if series else None
        if actual is None:
            continue
        event = {
            "event_id": f"verification:{fid}", "event_type": "verification",
            "forecast_event_id": fid, "station_id": forecast["station_id"],
            "target_at": forecast["target_at"], "horizon_days": forecast["horizon_days"],
            "predicted_aqi": forecast["predicted_aqi"], "actual_aqi": round(float(actual), 2),
            "absolute_error": round(abs(float(actual)-float(forecast["predicted_aqi"])), 2),
            "verified_at": anchor.isoformat(),
        }
        scored += int(_append_event(path, event, existing_ids))
    return {"forecast_events_added": added, "verification_events_added": scored}


def _category(value):
    return ("Good" if value <= 50 else "Satisfactory" if value <= 100 else
            "Moderate" if value <= 200 else "Poor" if value <= 300 else
            "Very Poor" if value <= 400 else "Severe")


def predict_remaining_month(station_id: int, *, issued_at=None,
                            timezone_name="Asia/Kolkata", history_path=None,
                            model_path=MODEL_PATH, report_path=REPORT_PATH,
                            additional_observations=None, max_data_age_hours=6.0):
    """Always return the remaining calendar dates; only validated horizons get values."""
    history_path = Path(history_path) if history_path else ROOT/"data"/"station_hourly_history.jsonl.gz"
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        artifact = joblib.load(model_path)
        if report.get("schema_version") != 1 or artifact.get("validation_status") != "validated":
            report, artifact = {}, {}
    except Exception:
        report, artifact = {}, {}

    sid = int(station_id)
    # The live dashboard forecasts one selected station. Stream the compressed
    # national history but retain only that station's rows in memory.
    records = read_history(history_path, station_id=sid)
    for item in additional_observations or []:
        try:
            records.append({"station_id": sid, "sensor_id": int(item["sensor_id"]),
                            "pollutant": str(item["pollutant"]).lower(),
                            "timestamp": item["timestamp"], "value": float(item["value"])})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    series = hourly_station_series(records).get(sid, {})
    now = _utc(issued_at or datetime.now(timezone.utc))
    anchor = now.replace(minute=0, second=0, microsecond=0)
    features = _features_at(series, anchor) if series else None

    fresh = []
    if features is not None:
        for pollutant in POLLUTANTS:
            window = {h: v for h, v in series.get(pollutant, {}).items()
                      if anchor-timedelta(hours=24) <= h < anchor}
            if len(window) < 16:
                continue
            if pollutant in {"co", "o3"}:
                ordered = sorted(window)
                complete = any(
                    len(ordered[i:i+8]) == 8
                    and all(ordered[i+j]-ordered[i+j-1] == timedelta(hours=1) for j in range(1, 8))
                    and ordered[i+7] >= anchor-timedelta(hours=max_data_age_hours)
                    for i in range(max(0, len(ordered)-7))
                )
                if not complete:
                    continue
            fresh.append(pollutant)
    recent = [h for p in fresh for h in series.get(p, {})
              if anchor-timedelta(hours=max_data_age_hours) <= h < anchor]
    age = (anchor-max(recent)).total_seconds()/3600 if recent else float("inf")
    data_ready = (
        features is not None and len(fresh) >= 3
        and bool({"pm25", "pm10"} & set(fresh))
        and 0 <= age <= max_data_age_hours
    )
    local_today = now.astimezone(ZoneInfo(timezone_name)).date()
    month_end = (local_today.replace(day=28)+timedelta(days=4)).replace(day=1)-timedelta(days=1)
    current_value = float(features["aqi_current"]) if features is not None else None
    days = [{
        "date": local_today.isoformat(), "horizon_days": 0,
        "aqi": round(current_value, 1) if current_value is not None else None,
        "lower_90": None, "upper_90": None,
        "status": "current_estimate" if current_value is not None else "current_unavailable",
        "category": _category(current_value) if current_value is not None else "Unavailable",
    }]
    cursor = local_today+timedelta(days=1)
    models = artifact.get("models", {})
    while cursor <= month_end:
        horizon = (cursor-local_today).days
        info = models.get(horizon) or models.get(str(horizon))
        gate = report.get("horizons", {}).get(str(horizon), {})
        if data_ready and info and info.get("validation_status") == "validated" and gate.get("forecast_enabled"):
            try:
                value = float(info["model"].predict(pd.DataFrame([features], columns=FEATURES))[0])
                if not math.isfinite(value):
                    raise ValueError("non-finite forecast")
                value = max(0, min(500, value))
                band = max(0, float(info.get("residual_p90", 0)))
                days.append({"date": cursor.isoformat(), "horizon_days": horizon,
                             "aqi": round(value, 1), "lower_90": round(max(0, value-band), 1),
                             "upper_90": round(min(500, value+band), 1),
                             "status": "validated_forecast", "category": _category(value)})
            except Exception:
                days.append({"date": cursor.isoformat(), "horizon_days": horizon,
                             "aqi": None, "status": "unavailable"})
        else:
            reason = (
                "Fresh station-hour coverage is not sufficient."
                if not data_ready else gate.get("status", "No validated model for this horizon.")
            )
            days.append({"date": cursor.isoformat(), "horizon_days": horizon,
                         "aqi": None, "status": "unvalidated", "reason": reason})
        cursor += timedelta(days=1)
    return {
        "available": True, "station_id": sid, "issued_at": anchor.isoformat(),
        "timezone": timezone_name, "data_age_hours": round(age, 2) if math.isfinite(age) else None,
        "data_ready": data_ready,
        "validated_horizons_days": report.get("validated_horizons_days", []),
        "days": days,
        "reason": None if data_ready else "Current station history has not met the freshness and pollutant coverage gates.",
        "basis": "OpenAQ-derived station-hour research estimates; not official CPCB AQI.",
    }

def main():
    records = read_history()
    ledger = {"forecast_events_added": 0, "verification_events_added": 0}
    # Score forecasts issued by a previous model before retraining can retire it.
    if MODEL_PATH.exists():
        try:
            ledger = update_forecast_ledger(records, joblib.load(MODEL_PATH))
        except Exception:
            pass
    report = train_monthly_models(records)
    if report.get("validated_horizons_days"):
        fresh_ledger = update_forecast_ledger(records, joblib.load(MODEL_PATH))
        ledger = {
            "forecast_events_added": ledger["forecast_events_added"] + fresh_ledger["forecast_events_added"],
            "verification_events_added": ledger["verification_events_added"] + fresh_ledger["verification_events_added"],
        }
    print(json.dumps({"report": report, "ledger": ledger}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
