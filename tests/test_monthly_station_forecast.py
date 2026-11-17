from datetime import datetime, timedelta, timezone
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import monthly_station_forecast as monthly
from station_forecast import FEATURES


def history(station_id=1, start=None, hours=240, offset=0):
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = {"pm25": 20+offset, "pm10": 40+offset, "no2": 20,
              "so2": 10, "co": .5, "o3": 30}
    rows = []
    for h in range(hours):
        stamp = start + timedelta(hours=h)
        for sensor, (pollutant, value) in enumerate(values.items(), 1):
            rows.append({"station_id": station_id, "sensor_id": sensor,
                         "pollutant": pollutant, "timestamp": stamp.isoformat(),
                         "value": value + (h % 12)*.05})
    return rows


def test_horizon_frame_uses_future_window_as_target_only():
    rows = history(hours=240)
    baseline = monthly.build_horizon_frame(rows, 2)
    assert not baseline.empty
    assert set(FEATURES).issubset(baseline.columns)
    assert (baseline["target_aqi"] >= 0).all()
    first = baseline.iloc[0]
    issue = pd.Timestamp(first.issued_at)
    altered = []
    for row in rows:
        item = dict(row)
        if pd.Timestamp(item["timestamp"]) >= issue:
            item["value"] *= 2.5
        altered.append(item)
    changed = monthly.build_horizon_frame(altered, 2)
    left = baseline[baseline.issued_at == first.issued_at].iloc[0]
    right = changed[changed.issued_at == first.issued_at].iloc[0]
    for feature in FEATURES:
        assert left[feature] == pytest.approx(right[feature], nan_ok=True)
    assert left.target_aqi != right.target_aqi


def test_horizon_validation_range_is_enforced():
    with pytest.raises(ValueError, match="between 1 and 30"):
        monthly.build_horizon_frame([], 0)
    with pytest.raises(ValueError, match="between 1 and 30"):
        monthly.build_horizon_frame([], 31)


def test_trainer_withholds_monthly_models_when_station_count_is_insufficient(tmp_path):
    records = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for station in range(1, 4):
        records.extend(history(station, start, 24*12, station))
    report = monthly.train_monthly_models(
        records, model_path=tmp_path/"monthly.joblib",
        report_path=tmp_path/"report.json", max_horizon_days=3,
    )
    assert report["status"] == "insufficient_data"
    assert report["validated_horizons_days"] == []
    assert not (tmp_path/"monthly.joblib").exists()
    assert report["horizons"]["1"]["forecast_enabled"] is False


def test_remaining_month_returns_all_dates_and_marks_unvalidated_horizons(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    rows = history(42, now-timedelta(hours=240), 241)
    path = tmp_path/"history.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    report = {
        "schema_version": 1,
        "validated_horizons_days": [1],
        "horizons": {"1": {"forecast_enabled": True, "status": "validated"}},
    }
    (tmp_path/"report.json").write_text(json.dumps(report), encoding="utf-8")

    class Model:
        def predict(self, frame):
            assert len(frame) == 1
            return [123.0]

    monkeypatch.setattr(monthly.joblib, "load", lambda _path: {
        "validation_status": "validated",
        "models": {1: {"model": Model(), "validation_status": "validated",
                       "residual_p90": 12}},
    })
    result = monthly.predict_remaining_month(
        42, issued_at=now, timezone_name="Asia/Kolkata",
        history_path=path, model_path=tmp_path/"model.joblib",
        report_path=tmp_path/"report.json",
    )
    assert result["available"] is True
    assert result["days"][0]["date"] == "2026-11-07"
    assert result["days"][1]["date"] == "2026-11-08"
    assert result["days"][1]["aqi"] == 123.0
    assert result["days"][1]["lower_90"] == 111.0
    assert result["days"][-1]["date"] == "2026-11-30"
    assert all(d["status"] == "unvalidated" for d in result["days"][2:])


def test_forecast_ledger_is_append_only_and_scores_only_matured_targets(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = history(7, start, 24*18)
    class Model:
        def predict(self, frame):
            return [80.0]
    artifact = {"trained_at": "v1", "models": {
        "1": {"model": Model(), "residual_p90": 10}
    }}
    ledger = tmp_path/"ledger.jsonl"
    issued = datetime(2026, 1, 5, tzinfo=timezone.utc)
    first = monthly.update_forecast_ledger(rows, artifact, ledger_path=ledger, now=issued)
    again = monthly.update_forecast_ledger(rows, artifact, ledger_path=ledger, now=issued)
    assert first["forecast_events_added"] == 1
    assert again["forecast_events_added"] == 0
    matured = monthly.update_forecast_ledger(
        rows, artifact, ledger_path=ledger, now=issued+timedelta(days=2)
    )
    assert matured["verification_events_added"] == 1
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    forecast = next(event for event in events if event["event_type"] == "forecast")
    verification = next(event for event in events if event["event_type"] == "verification")
    assert forecast["horizon_days"] == 1
    assert forecast["input_data_cutoff"] == issued.isoformat()
    assert verification["forecast_event_id"] == forecast["event_id"]
    assert verification["absolute_error"] >= 0
