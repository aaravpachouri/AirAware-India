"""Tests for nearby sensor-aware station discovery."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_air


def test_station_discovery_ranks_sensor_rich_nearby_candidates(monkeypatch):
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
        {"id": 1, "city": "Kanpur", "state": "Uttar Pradesh", "latitude": 26.45, "longitude": 80.33},
        {"id": 2, "city": "Kanpur", "state": "Uttar Pradesh", "latitude": 26.46, "longitude": 80.34, "name": "Better"},
        {"id": 3, "city": "Kanpur", "state": "Uttar Pradesh", "latitude": 26.47, "longitude": 80.35, "name": "Limited"},
        {"id": 4, "city": "Delhi", "state": "Delhi", "latitude": 28.61, "longitude": 77.20, "name": "Far away"},
    ]
    found = live_air.find_sensor_compatible_locations(locations, locations[0], max_candidates=10)
    assert {item["id"] for item in found} == {2, 3}
    assert found[0]["pollutants"] == ["no2", "pm25", "so2"]
    assert found[0]["inventory_eligible"] is True
    assert found[0]["distance_km"] < found[1]["distance_km"]


def test_station_discovery_works_when_city_metadata_is_missing(monkeypatch):
    monkeypatch.setattr(live_air, "get_location", lambda station_id: {"sensors": []})
    locations = [
        {"id": 1, "latitude": 19.076, "longitude": 72.877},
        {"id": 2, "latitude": 19.080, "longitude": 72.880},
    ]
    found = live_air.find_sensor_compatible_locations(locations, locations[0])
    assert len(found) == 1
    assert found[0]["id"] == 2
    assert found[0]["distance_km"] is not None


def test_station_discovery_returns_empty_without_city_or_coordinates():
    assert live_air.find_sensor_compatible_locations(
        [{"id": 2, "name": "Station"}], {"id": 1, "name": "Current"}
    ) == []


def test_station_discovery_bounds_metadata_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(live_air, "get_location", lambda station_id: calls.append(station_id) or {"sensors": []})
    locations = [
        {"id": station_id, "latitude": 19.0 + station_id * 0.0001, "longitude": 72.0}
        for station_id in range(1, 55)
    ]
    found = live_air.find_sensor_compatible_locations(locations, locations[0], max_candidates=1000, radius_km=50)
    assert len(found) == 40
    assert len(calls) == 40


def test_station_discovery_respects_radius(monkeypatch):
    monkeypatch.setattr(live_air, "get_location", lambda station_id: {"sensors": []})
    locations = [
        {"id": 1, "latitude": 19.0, "longitude": 72.0},
        {"id": 2, "latitude": 19.1, "longitude": 72.0},
        {"id": 3, "latitude": 20.0, "longitude": 72.0},
    ]
    found = live_air.find_sensor_compatible_locations(locations, locations[0], radius_km=20)
    assert [item["id"] for item in found] == [2]



def test_station_discovery_prefers_nearest_when_inventory_is_equal(monkeypatch):
    monkeypatch.setattr(live_air, "get_location", lambda station_id: {"sensors": [
        {"id": station_id * 10 + 1, "parameter": {"name": "pm2.5", "units": "µg/m³"}},
    ]})
    locations = [
        {"id": 1, "latitude": 19.0, "longitude": 72.0},
        {"id": 2, "latitude": 19.2, "longitude": 72.0, "last_datetime": "2026-11-10T01:00:00Z"},
        {"id": 3, "latitude": 19.05, "longitude": 72.0, "last_datetime": "2026-11-09T01:00:00Z"},
    ]
    found = live_air.find_sensor_compatible_locations(locations, locations[0], radius_km=30)
    assert [item["id"] for item in found] == [3, 2]


def test_station_discovery_keeps_sensor_inventory_ahead_of_distance(monkeypatch):
    details = {
        2: {"sensors": [{"id": 21, "parameter": {"name": "pm2.5", "units": "µg/m³"}}]},
        3: {"sensors": [
            {"id": 31, "parameter": {"name": "pm2.5", "units": "µg/m³"}},
            {"id": 32, "parameter": {"name": "no2", "units": "ppb"}},
            {"id": 33, "parameter": {"name": "so2", "units": "ppb"}},
        ]},
    }
    monkeypatch.setattr(live_air, "get_location", lambda station_id: details[station_id])
    locations = [
        {"id": 1, "latitude": 19.0, "longitude": 72.0},
        {"id": 2, "latitude": 19.01, "longitude": 72.0},
        {"id": 3, "latitude": 19.2, "longitude": 72.0},
    ]
    found = live_air.find_sensor_compatible_locations(locations, locations[0], radius_km=30)
    assert [item["id"] for item in found] == [3, 2]
