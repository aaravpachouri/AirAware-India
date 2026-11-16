"""Regression tests for full sensor inventory and hourly diagnostics."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import live_air


def test_coverage_check_refreshes_full_location_and_reports_missing_sensor(monkeypatch):
    now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
    full_location = {
        "name": "Mumbai test station",
        "sensors": [
            {"id": 101, "parameter": {"name": "pm2.5", "units": "µg/m³"}},
            {"id": 102, "parameter": {"name": "no2", "units": "ppb"}},
            {"id": 103, "parameter": {"name": "so2", "units": "ppb"}},
        ],
    }
    calls = []
    monkeypatch.setattr(live_air, "get_location", lambda location_id: full_location)

    def fake_get(endpoint, params=None):
        calls.append(endpoint)
        sensor_id = int(endpoint.split("/")[2])
        pollutant_value = {101: 20.0, 102: 10.0, 103: 5.0}[sensor_id]
        return {
            "results": [
                {
                    "datetimeFrom": {
                        "utc": (now - timedelta(hours=offset)).isoformat()
                    },
                    "value": pollutant_value,
                }
                for offset in range(1, 17)
            ]
        }

    monkeypatch.setattr(live_air, "_get", fake_get)
    # Deliberately pass a stale/partial station object. The function must
    # refresh full location details rather than trust this sensor list.
    partial = {"name": "Mumbai", "sensors": [
        {"id": 999, "parameter": {"name": "pm2.5", "units": "µg/m³"}}
    ]}
    result = live_air.get_cpcb_averaged_air_quality(
        77, partial, now=now
    )

    assert set(calls) == {
        "/sensors/101/hours", "/sensors/102/hours", "/sensors/103/hours"
    }
    assert result["sensor_diagnostics"]["pm10"]["sensor_count"] == 0
    assert "no sensor" in result["rejected_pollutants"]["pm10"]
    assert result["coverage"]["pm25"]["valid_hours_in_24h"] == 16


def test_hourly_request_failures_are_visible_in_sensor_diagnostics(monkeypatch):
    now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
    full_location = {
        "name": "Failure test station",
        "sensors": [
            {"id": 201, "parameter": {"name": "pm10", "units": "µg/m³"}}
        ],
    }
    monkeypatch.setattr(live_air, "get_location", lambda location_id: full_location)

    def failing_get(endpoint, params=None):
        raise RuntimeError("temporary provider outage")

    monkeypatch.setattr(live_air, "_get", failing_get)
    result = live_air.get_cpcb_averaged_air_quality(88, now=now)

    diagnostic = result["sensor_diagnostics"]["pm10"]
    assert diagnostic["sensor_count"] == 1
    assert diagnostic["requests_succeeded"] == 0
    assert "temporary provider outage" in diagnostic["errors"][0]
    assert "hourly request returned no records" in result["rejected_pollutants"]["pm10"]



def test_station_history_backfill_requests_14_days_and_preserves_raw_hourly_rows(monkeypatch):
    now = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
    full_location = {
        "name": "Backfill test station",
        "sensors": [
            {"id": 301, "parameter": {"name": "pm2.5", "units": "µg/m³"}}
        ],
    }
    captured = {}
    monkeypatch.setattr(live_air, "get_location", lambda location_id: full_location)

    def fake_get(endpoint, params=None):
        captured.update(params or {})
        rows = [{
            "datetimeFrom": {"utc": (now - timedelta(days=10)).isoformat()},
            "value": 18.0,
        }]
        rows.extend({
            "datetimeFrom": {"utc": (now - timedelta(hours=offset)).isoformat()},
            "value": 20.0 + offset / 10,
        } for offset in range(1, 17))
        return {"results": rows}

    monkeypatch.setattr(live_air, "_get", fake_get)
    result = live_air.get_cpcb_averaged_air_quality(
        99, now=now, history_days=14
    )

    assert captured["limit"] == 1000
    assert captured["datetime_from"] == (now - timedelta(days=14)).isoformat()
    assert len(result["hourly_observations"]) == 17
    assert any(row["timestamp"] == (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
               for row in result["hourly_observations"])
