"""Tests for opt-in sensor-aware station discovery."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_air


def test_station_discovery_ranks_sensor_rich_same_city_candidates(monkeypatch):
    details = {
        2: {"sensors": [
            {"id": 21, "parameter": {"name": "pm2.5", "units": "µg/m³"}},
            {"id": 22, "parameter": {"name": "no2", "units": "ppb"}},
            {"id": 23, "parameter": {"name": "so2", "units": "ppb"}},
        ]},
        3: {"sensors": [
            {"id": 31, "parameter": {"name": "pm2.5", "units": "µg/m³"}},
        ]},
    }
    monkeypatch.setattr(live_air, "get_location", lambda station_id: details[station_id])
    locations = [
        {"id": 1, "city": "Mumbai", "state": "Maharashtra", "name": "Current"},
        {"id": 2, "city": "Mumbai", "state": "Maharashtra", "name": "Better", "last_datetime": "2026-11-10"},
        {"id": 3, "city": "Mumbai", "state": "Maharashtra", "name": "Limited", "last_datetime": "2026-11-11"},
        {"id": 4, "city": "Pune", "state": "Maharashtra", "name": "Other city"},
    ]

    found = live_air.find_sensor_compatible_locations(
        locations, locations[0], max_candidates=10
    )

    assert [item["id"] for item in found] == [2, 3]
    assert found[0]["pollutants"] == ["no2", "pm25", "so2"]
    assert found[0]["inventory_eligible"] is True
    assert found[1]["inventory_eligible"] is False


def test_station_discovery_returns_empty_without_city_context():
    assert live_air.find_sensor_compatible_locations(
        [{"id": 2, "name": "Station"}],
        {"id": 1, "name": "Current"},
    ) == []


def test_station_discovery_bounds_metadata_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(
        live_air, "get_location",
        lambda station_id: calls.append(station_id) or {"sensors": []},
    )
    locations = [
        {"id": station_id, "city": "Mumbai", "state": "Maharashtra"}
        for station_id in range(1, 40)
    ]
    found = live_air.find_sensor_compatible_locations(
        locations, locations[0], max_candidates=1000
    )
    assert len(found) == 30
    assert len(calls) == 30
