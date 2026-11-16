"""Tests for append-only station-hour history storage."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from station_history import append_station_observations


def test_history_is_deduplicated_by_station_sensor_pollutant_and_hour(tmp_path):
    path = tmp_path / "history.jsonl"
    station = {"id": 11, "name": "Test station", "city": "Delhi"}
    rows = [{
        "pollutant": "pm25",
        "sensor_id": 101,
        "timestamp": "2026-11-16T10:00:00Z",
        "value": 25.5,
    }]
    first = append_station_observations(station, rows, path)
    second = append_station_observations(station, rows, path)
    assert first["records_added"] == 1
    assert second["records_added"] == 0
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(saved) == 1
    assert saved[0]["station_id"] == 11
    assert saved[0]["value"] == 25.5


def test_history_skips_invalid_values_and_unsupported_pollutants(tmp_path):
    path = tmp_path / "history.jsonl"
    rows = [
        {"pollutant": "pm10", "sensor_id": 1, "timestamp": "2026-11-16T10:00:00Z", "value": -1},
        {"pollutant": "temperature", "sensor_id": 2, "timestamp": "2026-11-16T10:00:00Z", "value": 20},
        {"pollutant": "no2", "sensor_id": 3, "timestamp": "2026-11-16T10:00:00Z", "value": float("nan")},
    ]
    result = append_station_observations({"id": 2}, rows, path)
    assert result["records_added"] == 0
    assert path.read_text() == ""


def test_history_normalizes_timestamps_to_utc_and_preserves_distinct_sensors(tmp_path):
    path = tmp_path / "history.jsonl"
    rows = [
        {"pollutant": "pm25", "sensor_id": 1, "timestamp": "2026-11-16T11:00:00+01:00", "value": 10},
        {"pollutant": "pm25", "sensor_id": 2, "timestamp": "2026-11-16T10:00:00Z", "value": 12},
    ]
    result = append_station_observations({"id": 3}, rows, path)
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert result["records_added"] == 2
    assert {row["timestamp"] for row in saved} == {"2026-11-16T10:00:00Z"}
    assert {row["sensor_id"] for row in saved} == {1, 2}
