"""Tests for CPCB AQI averaging-period and coverage gates."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_aqi_validation import assess_cpcb_aqi_eligibility


def test_latest_value_snapshot_is_not_eligible_without_coverage_metadata():
    result = assess_cpcb_aqi_eligibility(
        {"pm25": 42, "pm10": 70, "no2": 18, "o3": 25}, None
    )
    assert result["eligible"] is False
    assert result["eligible_pollutants"] == []
    assert result["status"] == "insufficient_validated_coverage"


def test_requires_three_pollutants_including_particulate_matter():
    values = {"pm25": 42, "no2": 18, "so2": 12}
    coverage = {
        key: {"averaging_period": "24h", "coverage_sufficient": True}
        for key in values
    }
    result = assess_cpcb_aqi_eligibility(values, coverage)
    assert result["eligible"] is True
    assert set(result["eligible_pollutants"]) == {"pm25", "no2", "so2"}


def test_gas_pollutants_require_eight_hour_averages():
    values = {"pm25": 42, "no2": 18, "o3": 25}
    coverage = {
        "pm25": {"averaging_period": "24h", "coverage_sufficient": True},
        "no2": {"averaging_period": "24h", "coverage_sufficient": True},
        "o3": {"averaging_period": "24h", "coverage_sufficient": True},
    }
    result = assess_cpcb_aqi_eligibility(values, coverage)
    assert result["eligible"] is False
    assert result["rejected_pollutants"]["o3"] == "requires 8h average"


def test_insufficient_coverage_rejects_pollutant():
    values = {"pm10": 70, "no2": 18, "so2": 12}
    coverage = {
        "pm10": {"averaging_period": "24h", "coverage_sufficient": True},
        "no2": {"averaging_period": "24h", "coverage_sufficient": False},
        "so2": {"averaging_period": "24h", "coverage_sufficient": True},
    }
    result = assess_cpcb_aqi_eligibility(values, coverage)
    assert result["eligible"] is False
    assert result["eligible_pollutants"] == ["pm10", "so2"]
