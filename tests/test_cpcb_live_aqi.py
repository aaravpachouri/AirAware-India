"""Coverage and averaging-window tests for live AQI aggregation."""
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpcb_live_aqi import aggregate_hourly_observations


NOW = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


def series(hours, value=20):
    return [
        {"timestamp": NOW - timedelta(hours=offset), "value": value + offset / 10}
        for offset in range(hours, 0, -1)
    ]


def test_24_hour_pollutant_requires_at_least_16_distinct_hours():
    result = aggregate_hourly_observations(
        {"pm25": series(15)}, now=NOW
    )
    assert "pm25" not in result["averages"]
    assert result["coverage"]["pm25"]["valid_hours_in_24h"] == 15
    assert result["status"] == "insufficient_validated_coverage"


def test_24_hour_average_is_calculated_when_coverage_is_sufficient():
    result = aggregate_hourly_observations(
        {"pm25": series(24)}, now=NOW
    )
    assert "pm25" in result["averages"]
    assert result["coverage"]["pm25"]["averaging_period"] == "24h"
    assert result["coverage"]["pm25"]["coverage_sufficient"] is True


def test_8_hour_pollutant_requires_a_complete_consecutive_8_hour_window():
    values = series(16)
    # Remove one point from the most recent eight-hour span while leaving
    # enough total hourly points in the 24-hour coverage window.
    values = [v for v in values if v["timestamp"] != NOW - timedelta(hours=3)]
    result = aggregate_hourly_observations({"co": values}, now=NOW)
    assert "co" not in result["averages"]
    assert "8-consecutive-hour" in result["rejected_pollutants"]["co"]


def test_duplicate_readings_do_not_inflate_hour_coverage():
    values = series(10)
    values.extend(values[:5])
    result = aggregate_hourly_observations({"pm10": values}, now=NOW)
    assert result["coverage"]["pm10"]["valid_hours_in_24h"] == 10
    assert "pm10" not in result["averages"]


def test_overall_eligibility_needs_three_pollutants_including_pm25_or_pm10():
    result = aggregate_hourly_observations(
        {
            "pm25": series(24, 20),
            "no2": series(24, 10),
            "so2": series(24, 5),
        },
        now=NOW,
    )
    assert result["eligible"] is True
    assert set(result["eligible_pollutants"]) == {"pm25", "no2", "so2"}


def test_negative_and_nonfinite_values_are_not_used():
    values = series(20)
    values[0]["value"] = -5
    values[1]["value"] = float("inf")
    result = aggregate_hourly_observations({"pm25": values}, now=NOW)
    assert result["coverage"]["pm25"]["valid_hours_in_24h"] == 18
