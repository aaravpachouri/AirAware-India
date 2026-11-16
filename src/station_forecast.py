"""Station-level, leakage-safe 24-hour AQI trend dataset and model.

All inputs are OpenAQ hourly sensor observations. The resulting index is an
OpenAQ-derived CPCB-style research estimate, not an official CPCB reading.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / "data" / "station_hourly_history.jsonl"
MODEL_PATH = ROOT / "models" / "airaware_station_trend.joblib"
REPORT_PATH = ROOT / "reports" / "station_model_validation.json"
POLLUTANTS = ("pm25", "pm10", "no2", "so2", "co", "o3")
LONG_WINDOW = {"pm25", "pm10", "no2", "so2"}
MIN_HOURS = 16
MIN_POLLUTANTS = 3

# Continuous CPCB-style breakpoints avoid artificial gaps between integer bands.
BREAKPOINTS = {
    "pm25": ((0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
             (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500)),
    "pm10": ((0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
             (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500)),
    "no2": ((0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200),
            (180, 280, 200, 300), (280, 400, 300, 400), (400, 1000, 400, 500)),
    "so2": ((0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200),
            (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2000, 400, 500)),
    "co": ((0, 1, 0, 50), (1, 2, 50, 100), (2, 10, 100, 200),
           (10, 17, 200, 300), (17, 34, 300, 400), (34, 50, 400, 500)),
    "o3": ((0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200),
           (168, 208, 200, 300), (208, 748, 300, 400), (748, 1000, 400, 500)),
}

FEATURES = (
    "aqi_current", "aqi_lag_24h", "aqi_lag_48h", "aqi_3day_mean",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "pm25_change_24h", "pm10_change_24h", "no2_change_24h",
    "so2_change_24h", "co_change_24h", "o3_change_24h",
)


def _utc(value: Any) -> datetime:
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def read_history(path: Path = DEFAULT_HISTORY) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                station_id = int(row["station_id"])
                sensor_id = int(row["sensor_id"])
                pollutant = str(row["pollutant"]).lower()
                value = float(row["value"])
                stamp = _utc(row["timestamp"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid station history at line {line_no}") from exc
            if pollutant not in POLLUTANTS or not math.isfinite(value) or value < 0:
                continue
            rows.append({**row, "station_id": station_id, "sensor_id": sensor_id,
                         "pollutant": pollutant, "value": value, "timestamp": stamp})
    return rows


def hourly_station_series(records: list[dict]) -> dict[int, dict[str, dict[datetime, float]]]:
    """Latest reading per sensor/hour, then mean co-located sensors per station/hour."""
    sensor_hour: dict[tuple[int, str, datetime, int], tuple[datetime, float]] = {}
    for row in records:
        try:
            station = int(row["station_id"])
            sensor = int(row["sensor_id"])
            pollutant = str(row["pollutant"]).lower()
            stamp = _utc(row["timestamp"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if pollutant not in POLLUTANTS or not math.isfinite(value) or value < 0:
            continue
        hour = stamp.replace(minute=0, second=0, microsecond=0)
        key = (station, pollutant, hour, sensor)
        previous = sensor_hour.get(key)
        if previous is None or stamp >= previous[0]:
            sensor_hour[key] = (stamp, value)
    grouped: dict[tuple[int, str, datetime], list[float]] = defaultdict(list)
    for (station, pollutant, hour, _sensor), (_stamp, value) in sensor_hour.items():
        grouped[(station, pollutant, hour)].append(value)
    output: dict[int, dict[str, dict[datetime, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (station, pollutant, hour), values in grouped.items():
        output[station][pollutant][hour] = float(np.mean(values))
    return output


def _subindex(pollutant: str, concentration: float) -> float | None:
    if not math.isfinite(concentration) or concentration < 0:
        return None
    for low, high, index_low, index_high in BREAKPOINTS[pollutant]:
        if low <= concentration <= high:
            if high == low:
                return float(index_high)
            return index_low + (concentration - low) * (index_high - index_low) / (high - low)
    return 500.0 if concentration > BREAKPOINTS[pollutant][-1][1] else None


def aqi_for_window(series: dict[str, dict[datetime, float]], anchor: datetime) -> tuple[float | None, dict[str, float]]:
    """AQI from [anchor-24h, anchor), with coverage gates and 8h CO/O3 means."""
    anchor = _utc(anchor).replace(minute=0, second=0, microsecond=0)
    start = anchor - timedelta(hours=24)
    concentrations: dict[str, float] = {}
    subindices: dict[str, float] = {}
    for pollutant in POLLUTANTS:
        values = {hour: value for hour, value in series.get(pollutant, {}).items()
                  if start <= hour < anchor}
        if len(values) < MIN_HOURS:
            continue
        if pollutant in LONG_WINDOW:
            concentration = float(np.mean(list(values.values())))
        else:
            ordered = sorted(values.items())
            candidate = None
            for index in range(len(ordered) - 8, -1, -1):
                block = ordered[index:index + 8]
                if len(block) == 8 and all(
                    block[j][0] - block[j - 1][0] == timedelta(hours=1)
                    for j in range(1, 8)
                ):
                    candidate = float(np.mean([item[1] for item in block]))
                    break
            if candidate is None:
                continue
            concentration = candidate
        index = _subindex(pollutant, concentration)
        if index is not None:
            concentrations[pollutant] = concentration
            subindices[pollutant] = index
    if len(subindices) < MIN_POLLUTANTS or not ({"pm25", "pm10"} & subindices.keys()):
        return None, concentrations
    return float(max(subindices.values())), concentrations


def _features_at(series: dict[str, dict[datetime, float]], anchor: datetime) -> dict | None:
    current, concentrations = aqi_for_window(series, anchor)
    previous, _ = aqi_for_window(series, anchor - timedelta(hours=24))
    previous2, _ = aqi_for_window(series, anchor - timedelta(hours=48))
    if current is None or previous is None or previous2 is None:
        return None
    hour = anchor.hour + anchor.minute / 60
    weekday = anchor.weekday()
    row = {
        "aqi_current": current,
        "aqi_lag_24h": previous,
        "aqi_lag_48h": previous2,
        "aqi_3day_mean": float(np.mean([current, previous, previous2])),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday_sin": math.sin(2 * math.pi * weekday / 7),
        "weekday_cos": math.cos(2 * math.pi * weekday / 7),
    }
    for pollutant in POLLUTANTS:
        row[pollutant] = concentrations.get(pollutant, np.nan)
        old = None
        # Concentration trend compares two non-overlapping 24h windows.
        _old_aqi, old_values = aqi_for_window(series, anchor - timedelta(hours=24))
        old = old_values.get(pollutant)
        row[f"{pollutant}_change_24h"] = (
            concentrations[pollutant] - old
            if pollutant in concentrations and old is not None else np.nan
        )
    return row


def build_training_frame(records: list[dict]) -> pd.DataFrame:
    """Create issue-time rows with targets exactly 24h later; no future features."""
    stations = hourly_station_series(records)
    rows = []
    for station_id, pollutants in stations.items():
        all_hours = sorted({hour for values in pollutants.values() for hour in values})
        if not all_hours:
            continue
        first = all_hours[0].replace(minute=0, second=0, microsecond=0)
        last = all_hours[-1].replace(minute=0, second=0, microsecond=0)
        anchor = first + timedelta(hours=48)
        while anchor + timedelta(hours=24) <= last + timedelta(hours=1):
            features = _features_at(pollutants, anchor)
            target, _ = aqi_for_window(pollutants, anchor + timedelta(hours=24))
            if features is not None and target is not None:
                rows.append({
                    "station_id": station_id,
                    "issued_at": anchor.isoformat(),
                    **features,
                    "target_aqi_24h": target,
                })
            anchor += timedelta(hours=1)
    if not rows:
        return pd.DataFrame(columns=["station_id", "issued_at", *FEATURES, "target_aqi_24h"])
    return pd.DataFrame(rows).sort_values(["issued_at", "station_id"]).reset_index(drop=True)


def _metrics(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    actual = frame["target_aqi_24h"].to_numpy(dtype=float)
    persistence = frame["aqi_current"].to_numpy(dtype=float)
    rolling = frame["aqi_3day_mean"].to_numpy(dtype=float)
    model_mae = float(mean_absolute_error(actual, predictions))
    persistence_mae = float(mean_absolute_error(actual, persistence))
    rolling_mae = float(mean_absolute_error(actual, rolling))
    best_baseline = min(persistence_mae, rolling_mae)
    return {
        "samples": int(len(frame)),
        "stations": int(frame["station_id"].nunique()),
        "model_mae": round(model_mae, 3),
        "persistence_mae": round(persistence_mae, 3),
        "three_day_mean_mae": round(rolling_mae, 3),
        "best_baseline_mae": round(best_baseline, 3),
        "improvement_over_best_baseline_pct": round(
            (best_baseline - model_mae) / best_baseline * 100, 3
        ) if best_baseline > 0 else None,
    }


def train_and_validate(records: list[dict], *, model_path: Path = MODEL_PATH,
                       report_path: Path = REPORT_PATH) -> dict:
    """Train only when both chronological and station-held-out tests beat baselines."""
    frame = build_training_frame(records)
    stations = sorted(frame["station_id"].unique().tolist()) if not frame.empty else []
    if len(frame) < 500 or len(stations) < 5:
        report = {
            "status": "insufficient_data", "samples": int(len(frame)),
            "stations": len(stations), "minimum_samples": 500,
            "minimum_stations": 5,
            "reason": "Collect more real station-hour history before fitting a station model.",
        }
        _write_report(report, report_path)
        if Path(model_path).exists():
            Path(model_path).unlink()
        return report

    frame["issued_at"] = pd.to_datetime(frame["issued_at"], utc=True)
    split_time = frame["issued_at"].quantile(0.8)
    temporal_train = frame[frame["issued_at"] < split_time]
    temporal_test = frame[frame["issued_at"] >= split_time]
    heldout_ids = set(stations[-max(1, int(math.ceil(len(stations) * 0.2))):])
    station_train = frame[~frame["station_id"].isin(heldout_ids)]
    station_test = frame[frame["station_id"].isin(heldout_ids)]

    if min(len(temporal_train), len(temporal_test), len(station_train), len(station_test)) < 50:
        report = {
            "status": "insufficient_validation_data", "samples": len(frame),
            "stations": len(stations), "reason": "Not enough independent rows in both holdouts.",
        }
        report["forecast_enabled"] = False
        _write_report(report, report_path)
        if Path(model_path).exists():
            Path(model_path).unlink()
        return report

    def pipeline():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=0.9,
                random_state=42, n_jobs=-1,
            )),
        ])

    model = pipeline()
    model.fit(temporal_train[list(FEATURES)], temporal_train["target_aqi_24h"])
    temporal_metrics = _metrics(
        temporal_test, model.predict(temporal_test[list(FEATURES)])
    )

    # A second model is fitted without any samples from held-out stations.
    station_model = pipeline()
    station_model.fit(station_train[list(FEATURES)], station_train["target_aqi_24h"])
    station_metrics = _metrics(
        station_test, station_model.predict(station_test[list(FEATURES)])
    )
    temporal_pass = temporal_metrics["model_mae"] < temporal_metrics["best_baseline_mae"] * 0.95
    station_pass = station_metrics["model_mae"] < station_metrics["best_baseline_mae"] * 0.95
    passed = bool(temporal_pass and station_pass)

    report = {
        "schema_version": 1,
        "status": "validated" if passed else "baseline_not_beaten",
        "source": "OpenAQ v3 station hourly sensor history",
        "target": "AQI derived from the following 24-hour station window",
        "forecast_horizon_hours": 24,
        "feature_columns": list(FEATURES),
        "samples": int(len(frame)),
        "stations": len(stations),
        "temporal_split_utc": split_time.isoformat(),
        "temporal_holdout": temporal_metrics,
        "station_holdout_ids": sorted(heldout_ids),
        "station_holdout": station_metrics,
        "minimum_required_improvement_pct": 5.0,
        "temporal_gate_passed": bool(temporal_pass),
        "station_gate_passed": bool(station_pass),
        "forecast_enabled": passed,
        "note": "Research estimate only; not official CPCB AQI or a regulatory 24-hour CPCB average.",
    }
    _write_report(report, report_path)
    model_path = Path(model_path)
    if passed:
        # Refit the approved architecture on all available history only after
        # both untouched evaluation splits have passed their baseline gates.
        final_model = pipeline()
        final_model.fit(frame[list(FEATURES)], frame["target_aqi_24h"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final_model, "features": list(FEATURES),
                     "validation_status": "validated", "report": str(report_path)}, model_path)
    elif model_path.exists():
        model_path.unlink()
    return report


def _write_report(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)



def predict_station_forecast(station_id: int, *, issued_at: datetime | None = None,
                             history_path: Path = DEFAULT_HISTORY,
                             model_path: Path = MODEL_PATH,
                             report_path: Path = REPORT_PATH,
                             max_data_age_hours: float = 6.0) -> dict:
    """Return a forecast only when the approved model and fresh station history exist."""
    report_path = Path(report_path)
    model_path = Path(model_path)
    if not report_path.exists() or not model_path.exists():
        return {"available": False, "reason": "No validated station-level model is published."}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"available": False, "reason": "Station model validation report is unavailable."}
    if report.get("status") != "validated" or report.get("forecast_enabled") is not True:
        return {"available": False, "reason": "Station model has not passed both baseline gates."}
    records = read_history(history_path)
    series_by_station = hourly_station_series(records)
    station_id = int(station_id)
    series = series_by_station.get(station_id)
    if not series:
        return {"available": False, "reason": "No hourly history is available for this station."}
    anchor = _utc(issued_at or datetime.now(timezone.utc))
    anchor = anchor.replace(minute=0, second=0, microsecond=0)
    features = _features_at(series, anchor)
    if features is None:
        return {"available": False, "reason": "Station history does not yet meet the 72-hour feature coverage requirement."}
    observed_hours = [
        hour for pollutant in POLLUTANTS
        for hour in series.get(pollutant, {})
        if hour < anchor
    ]
    if not observed_hours:
        return {"available": False, "reason": "No timestamped station measurements are available."}
    age = (anchor - max(observed_hours)).total_seconds() / 3600
    if age < 0 or age > float(max_data_age_hours):
        return {"available": False, "reason": "Station history is stale for a live forecast.", "data_age_hours": round(age, 2)}
    try:
        artifact = joblib.load(model_path)
        if artifact.get("validation_status") != "validated" or artifact.get("features") != list(FEATURES):
            return {"available": False, "reason": "Station model artifact failed its validation/schema check."}
        prediction = float(artifact["model"].predict(pd.DataFrame([features], columns=FEATURES))[0])
    except Exception as exc:
        return {"available": False, "reason": f"Station model could not produce a forecast: {str(exc)[:160]}"}
    if not math.isfinite(prediction):
        return {"available": False, "reason": "Station model returned a non-finite forecast."}
    return {
        "available": True,
        "station_id": station_id,
        "issued_at": anchor.isoformat().replace("+00:00", "Z"),
        "target_at": (anchor + timedelta(hours=24)).isoformat().replace("+00:00", "Z"),
        "current_aqi": round(float(features["aqi_current"]), 1),
        "predicted_aqi": round(max(0.0, min(500.0, prediction)), 1),
        "data_age_hours": round(age, 2),
        "validation": "Chronological and held-out-station tests beat persistence and 3-day mean by at least 5%.",
        "basis": "OpenAQ-derived station-hour history; not an official CPCB forecast.",
    }


def main() -> None:
    records = read_history()
    report = train_and_validate(records)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
