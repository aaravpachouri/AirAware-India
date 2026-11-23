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


def test_read_history_can_filter_to_selected_station(tmp_path):
    path = tmp_path / "history.jsonl.gz"
    import gzip
    rows = history(station_id=10, hours=4) + history(station_id=20, hours=4)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    selected = monthly.read_history(path, station_id=20)
    assert selected
    assert {row["station_id"] for row in selected} == {20}
    assert len(selected) == 4 * 6
    assert len(monthly.read_history(path)) == 8 * 6


def test_horizon_frame_uses_future_window_as_target_only():
    rows = history(hours=240)
    baseline = monthly.build_horizon_frame(rows, 2)
    assert not baseline.empty
    assert set(FEATURES).issubset(baseline.columns)
    assert (baseline["target_aqi"] >= 0).all()
    assert set(pd.to_datetime(baseline["issued_at"], utc=True).dt.hour) == set(range(24))
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


def test_horizon_frame_reuses_shared_series_and_caches_without_changing_values():
    rows = history(hours=240)
    series_map = monthly.hourly_station_series(rows)
    feature_cache, target_cache = {}, {}
    cached = monthly.build_horizon_frame(
        rows, 2, series_map=series_map,
        feature_cache=feature_cache, target_cache=target_cache,
    )
    uncached = monthly.build_horizon_frame(rows, 2)
    pd.testing.assert_frame_equal(cached, uncached)
    assert feature_cache
    assert target_cache


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



def test_month_calendar_is_rendered_even_before_any_horizon_is_validated(tmp_path):
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    rows = history(42, now-timedelta(hours=240), 241)
    path = tmp_path/"history.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    result = monthly.predict_remaining_month(
        42, issued_at=now, timezone_name="Asia/Kolkata",
        history_path=path, model_path=tmp_path/"missing.joblib",
        report_path=tmp_path/"missing-report.json",
    )
    assert result["available"] is True
    assert len(result["days"]) == 24
    assert result["days"][0]["date"] == "2026-11-07"
    assert result["days"][-1]["date"] == "2026-11-30"
    assert all(day["aqi"] is None for day in result["days"][1:])
    assert all(day["status"] == "unvalidated" for day in result["days"][1:])



def test_monthly_report_serializes_numpy_station_ids(tmp_path, monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    records = []
    for station_id in range(1, 51):
        records.extend([
            {"station_id": station_id, "timestamp": (start - pd.Timedelta(days=1)).isoformat()},
            {"station_id": station_id, "timestamp": (start + pd.Timedelta(days=40)).isoformat()},
        ])
        for day in range(10):
            for hour in range(24):
                issued = start + pd.Timedelta(days=day, hours=hour)
                rows.append({
                    "station_id": station_id,
                    "issued_at": issued.isoformat(),
                    **{feature: 50.0 for feature in FEATURES},
                    "target_aqi": 55.0,
                })
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(monthly, "build_horizon_frame", lambda _records, _horizon, **_kwargs: frame.copy())

    class DummyModel:
        def fit(self, _x, _y):
            return self

        def predict(self, x):
            return [50.0] * len(x)

    monkeypatch.setattr(monthly, "_pipeline", DummyModel)
    monkeypatch.setattr(monthly, "_metrics", lambda _frame, _prediction: {
        "samples": len(_frame), "stations": int(_frame.station_id.nunique()),
        "model_mae": 1.0, "persistence_mae": 10.0,
        "three_day_mean_mae": 9.0, "best_baseline_mae": 9.0,
        "improvement_over_best_baseline_pct": 88.889,
        "absolute_error_p90": 1.0,
    })

    # This regression targets JSON report serialization, not sklearn/joblib pickling.
    # The local dummy estimator is intentionally not pickleable.
    monkeypatch.setattr(monthly.joblib, "dump", lambda *_args, **_kwargs: None)

    report = monthly.train_monthly_models(
        records,
        model_path=tmp_path / "monthly.joblib",
        report_path=tmp_path / "report.json",
        max_horizon_days=1,
    )

    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["horizons"]["1"]["status"] == "validated"
    assert all(type(value) is int for value in saved["horizons"]["1"]["station_holdout_ids"])
