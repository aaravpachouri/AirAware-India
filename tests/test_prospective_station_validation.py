"""Deterministic tests for the prospective validation ledger."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prospective_station_validation import forecast_id, read_ledger, write_ledger, make_record, pair_observations
from evaluate_prospective_station_validation import summarize


def test_forecast_id_is_station_and_utc_day_scoped():
    issued = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    assert forecast_id(42, issued) == "openaq:42:2026-11-10"


def test_pairing_requires_same_station_and_18_to_30_hour_horizon():
    issued = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    record = {"status": "pending", "station": {"id": 42}, "issued_at": issued.isoformat(),
              "predicted_aqi": 120, "persistence_aqi": 100}
    observed = {"timestamp": issued + timedelta(hours=24), "aqi": 90,
                "pollutants": ["pm25", "pm10", "no2"]}
    assert pair_observations([record], {99: observed}, issued + timedelta(hours=24)) == 0
    assert record["status"] == "pending"
    assert pair_observations([record], {42: observed}, issued + timedelta(hours=24)) == 1
    assert record["status"] == "evaluated"
    assert record["absolute_error"] == 30
    assert record["persistence_absolute_error"] == 10


def test_pairing_rejects_observation_outside_horizon():
    issued = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    record = {"status": "pending", "station": {"id": 42}, "issued_at": issued.isoformat(),
              "predicted_aqi": 120, "persistence_aqi": 100}
    observed = {"timestamp": issued + timedelta(hours=12), "aqi": 90, "pollutants": ["pm25", "pm10", "no2"]}
    assert pair_observations([record], {42: observed}, issued + timedelta(hours=12)) == 0
    assert record["status"] == "pending"


def test_jsonl_ledger_round_trip(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    records = [{"forecast_id": "x"}, {"forecast_id": "y"}]
    write_ledger(records, path)
    assert read_ledger(path) == records
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_record_contains_model_and_station_provenance():
    issued = datetime(2026, 11, 10, 5, tzinfo=timezone.utc)
    data = {"location_id": 7, "location": "Station A", "city": "Delhi",
            "timestamp": issued - timedelta(hours=1), "age_hours": 1,
            "pollutants": {"pm25": 20}}
    record = make_record({"id": 7}, data, 101.4, 90, ["pm25", "pm10", "no2"], issued, "abc")
    assert record["station"]["id"] == 7
    assert record["predicted_aqi"] == 101.4
    assert record["model_sha256"] == "abc"
    assert record["status"] == "pending"
    assert record["forecast_horizon_hours"] == 24


def test_empty_summary_makes_no_accuracy_claim():
    result = summarize(pd.DataFrame())
    assert result["evaluated_forecasts"] == 0
    assert result["model_mae"] is None
    assert "No prospective" in result["interpretation"]


def _evaluated_row(station_id, city, model_error, baseline_error, observed=100, horizon=24):
    return {
        "absolute_error": model_error,
        "persistence_absolute_error": baseline_error,
        "station.id": station_id,
        "station.city": city,
        "station.name": f"Station {station_id}",
        "predicted_aqi": observed + model_error,
        "observed_aqi": observed,
        "actual_horizon_hours": horizon,
        "issued_at": "2026-11-10T00:00:00Z",
        "observed_at": "2026-11-11T00:00:00Z",
    }


def test_summary_reports_paired_metrics_and_station_coverage():
    frame = pd.DataFrame([
        _evaluated_row(1, "Delhi", 10, 15),
        _evaluated_row(2, "Delhi", 20, 15),
        _evaluated_row(3, "Pune", 5, 10),
    ])
    result = summarize(frame)
    assert result["evaluated_forecasts"] == 3
    assert result["stations"] == 3
    assert result["cities"] == 2
    assert result["model_mae"] == round((10 + 20 + 5) / 3, 3)
    assert result["persistence_mae"] == round((15 + 15 + 10) / 3, 3)
    assert result["mae_improvement"] == round((15 + 15 + 10 - 10 - 20 - 5) / 3, 3)
    assert result["station_cluster_bootstrap_95pct_ci_for_mae_improvement"] is None
    assert len(result["per_station"]) == 3


def test_summary_bootstraps_whole_station_clusters():
    frame = pd.DataFrame([
        _evaluated_row(1, "Delhi", 10, 20),
        _evaluated_row(1, "Delhi", 12, 22),
        _evaluated_row(2, "Pune", 15, 20),
        _evaluated_row(3, "Mumbai", 8, 14),
        _evaluated_row(4, "Chennai", 11, 16),
        _evaluated_row(5, "Kolkata", 9, 15),
    ])
    result = summarize(frame)
    interval = result["station_cluster_bootstrap_95pct_ci_for_mae_improvement"]
    assert interval is not None
    assert interval["lower"] <= interval["upper"]
    assert "Exploratory paired percentile interval" in result["bootstrap_note"]


def test_station_batches_rotate_without_permanent_newest_station_bias():
    from prospective_station_validation import select_station_batch

    stations = [{"id": value} for value in [9, 2, 7, 1, 5, 3, 8, 4, 6]]
    first, cursor = select_station_batch(stations, 4, None)
    second, cursor = select_station_batch(stations, 4, cursor)
    third, cursor = select_station_batch(stations, 4, cursor)

    assert [item["id"] for item in first] == [1, 2, 3, 4]
    assert [item["id"] for item in second] == [5, 6, 7, 8]
    assert [item["id"] for item in third] == [9, 1, 2, 3]


def test_station_batch_resumes_after_cursor_station_disappears():
    from prospective_station_validation import select_station_batch

    stations = [{"id": 2}, {"id": 8}, {"id": 12}, {"id": 20}]
    selected, next_cursor = select_station_batch(stations, 2, 9)
    assert [item["id"] for item in selected] == [12, 20]
    assert next_cursor == 2


def test_station_batch_handles_empty_inventory_and_cursor_round_trip(tmp_path):
    from prospective_station_validation import (
        read_station_cursor, select_station_batch, write_station_cursor,
    )

    assert select_station_batch([], 10, None) == ([], None)
    path = tmp_path / "cursor.json"
    assert read_station_cursor(path) is None
    write_station_cursor(42, path)
    assert read_station_cursor(path) == 42



def test_history_and_forecast_station_cursors_can_rotate_independently(tmp_path):
    from prospective_station_validation import (
        read_station_cursor, write_station_cursor, select_station_batch,
    )

    stations = [{"id": value} for value in range(1, 31)]
    forecast_path = tmp_path / "forecast-cursor.json"
    history_path = tmp_path / "history-cursor.json"

    forecast_batch, forecast_next = select_station_batch(stations, 25, None)
    history_batch, history_next = select_station_batch(stations, 10, None)
    write_station_cursor(forecast_next, forecast_path)
    write_station_cursor(history_next, history_path)

    assert [row["id"] for row in forecast_batch] == list(range(1, 26))
    assert [row["id"] for row in history_batch] == list(range(1, 11))
    assert read_station_cursor(forecast_path) == 26
    assert read_station_cursor(history_path) == 11

    next_history_batch, next_history = select_station_batch(
        stations, 10, read_station_cursor(history_path)
    )
    assert [row["id"] for row in next_history_batch] == list(range(11, 21))
    assert next_history == 21
