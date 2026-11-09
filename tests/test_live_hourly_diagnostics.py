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
