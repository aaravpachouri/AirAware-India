"""Tests for leakage-safe station-level forecasting data and validation gates."""
from datetime import datetime, timedelta, timezone
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from station_forecast import (
    BREAKPOINTS, FEATURES, aqi_for_window, build_training_frame,
    hourly_station_series, train_and_validate,
)


def _synthetic_history(station_id=1, start=None, hours=240, offset=0):
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    values = {"pm25": 20 + offset, "pm10": 40 + offset, "no2": 20,
              "so2": 10, "co": 0.5, "o3": 30}
    for hour in range(hours):
        stamp = start + timedelta(hours=hour)
        for sensor, (pollutant, value) in enumerate(values.items(), 1):
            rows.append({
                "station_id": station_id, "sensor_id": sensor,
                "pollutant": pollutant, "timestamp": stamp.isoformat(),
                "value": value + (hour % 12) * 0.05,
            })
    return rows


def test_sensor_values_are_deduplicated_then_averaged_per_station_hour():
    stamp = "2026-01-01T00:15:00Z"
    rows = [
        {"station_id": 1, "sensor_id": 1, "pollutant": "pm25", "timestamp": stamp, "value": 10},
        {"station_id": 1, "sensor_id": 1, "pollutant": "pm25", "timestamp": "2026-01-01T00:45:00Z", "value": 20},
        {"station_id": 1, "sensor_id": 2, "pollutant": "pm25", "timestamp": stamp, "value": 30},
    ]
    series = hourly_station_series(rows)
    hour = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert series[1]["pm25"][hour] == 25


def test_aqi_requires_coverage_and_at_least_three_pollutants_including_pm():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    series = {"pm25": {start + timedelta(hours=i): 20 for i in range(16)},
              "pm10": {start + timedelta(hours=i): 40 for i in range(16)}}
    aqi, _ = aqi_for_window(series, start + timedelta(hours=24))
    assert aqi is None


def test_fractional_breakpoint_gaps_are_interpolated_continuously():
    # PM2.5 values between old integer-only bands now have a defined index.
    assert 50 < station_subindex("pm25", 30.5) < 51


def station_subindex(pollutant, value):
    from station_forecast import _subindex
    return _subindex(pollutant, value)


def test_training_rows_use_future_window_only_as_target():
    rows = _synthetic_history(hours=180)
    frame = build_training_frame(rows)
    assert not frame.empty
    assert set(FEATURES).issubset(frame.columns)
    assert "target_aqi_24h" in frame.columns
    # First complete sample requires two prior 24h AQI windows.
    first_issue = datetime.fromisoformat(frame.iloc[0]["issued_at"])
    assert first_issue >= datetime(2026, 1, 3, tzinfo=timezone.utc)
    assert (frame["target_aqi_24h"] >= 0).all()


def test_dataset_does_not_make_rows_when_history_is_short():
    frame = build_training_frame(_synthetic_history(hours=30))
    assert frame.empty


def test_trainer_refuses_to_publish_model_without_real_station_coverage(tmp_path):
    report = train_and_validate(
        _synthetic_history(hours=180),
        model_path=tmp_path / "model.joblib",
        report_path=tmp_path / "report.json",
    )
    assert report["status"] == "insufficient_data"
    assert report["forecast_enabled"] is False
    assert not (tmp_path / "model.joblib").exists()


def test_invalid_or_nonfinite_observations_are_ignored():
    rows = _synthetic_history(hours=5)
    rows.extend([
        {"station_id": 1, "sensor_id": 99, "pollutant": "pm25",
         "timestamp": "2026-01-01T01:00:00Z", "value": float("nan")},
        {"station_id": 1, "sensor_id": 99, "pollutant": "pm25",
         "timestamp": "2026-01-01T01:00:00Z", "value": -1},
    ])
    series = hourly_station_series(rows)
    assert all(math.isfinite(v) and v >= 0 for v in series[1]["pm25"].values())



def test_future_observations_change_target_but_never_issue_time_features():
    rows = _synthetic_history(hours=180)
    baseline = build_training_frame(rows)
    assert not baseline.empty
    issue = baseline.iloc[0]["issued_at"]
    changed = []
    for row in rows:
        copy = dict(row)
        if datetime.fromisoformat(copy["timestamp"]) >= datetime.fromisoformat(issue):
            copy["value"] = float(copy["value"]) * 3
        changed.append(copy)
    altered = build_training_frame(changed)
    base_row = baseline[baseline["issued_at"] == issue].iloc[0]
    altered_row = altered[altered["issued_at"] == issue].iloc[0]
    for feature in FEATURES:
        left, right = base_row[feature], altered_row[feature]
        if math.isnan(float(left)):
            assert math.isnan(float(right))
        else:
            assert left == right
    assert base_row["target_aqi_24h"] != altered_row["target_aqi_24h"]



def test_training_issues_cover_four_six_hourly_times_per_day():
    frame = build_training_frame(_synthetic_history(hours=240))
    assert not frame.empty
    hours = set(pd.to_datetime(frame["issued_at"], utc=True).dt.hour)
    assert hours == {0, 6, 12, 18}
    ordered = pd.to_datetime(frame["issued_at"], utc=True).drop_duplicates().sort_values()
    assert set(ordered.diff().dropna().dt.total_seconds() / 3600) == {6.0}
