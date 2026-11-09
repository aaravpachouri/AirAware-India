"""Regression tests for OpenAQ unit normalization and AQI inputs."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_air import _convert_value, calculate_subindex, estimate_cpcb_style_aqi


@pytest.mark.parametrize(
    ("pollutant", "value", "unit", "expected"),
    [
        ("pm25", 25.0, "µg/m³", 25.0),
        ("pm10", 0.05, "mg/m³", 50.0),
        ("no2", 10.0, "ppb", 18.8),
        ("so2", 1.0, "ppm", 2620.0),
        ("o3", 50.0, "ppb", 98.0),
        ("co", 1000.0, "µg/m³", 1.0),
        ("co", 1.0, "ppm", 1.145),
        ("co", 1000.0, "ppb", 1.145),
    ],
)
def test_convert_value_to_cpcb_breakpoint_units(pollutant, value, unit, expected):
    assert _convert_value(pollutant, value, unit) == pytest.approx(expected)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_convert_value_rejects_invalid_concentrations(value):
    with pytest.raises(ValueError, match="Invalid concentration"):
        _convert_value("pm25", value, "µg/m³")


def test_convert_value_rejects_unknown_units_instead_of_assuming_compatibility():
    with pytest.raises(ValueError, match="Unsupported OpenAQ unit"):
        _convert_value("o3", 10, "unknown-unit")


def test_aqi_estimate_uses_highest_available_pollutant_subindex():
    result = estimate_cpcb_style_aqi({"pm25": 60, "pm10": 50, "no2": None})
    assert result["aqi"] == round(result["subindices"]["pm25"])
    assert result["aqi"] >= result["subindices"]["pm10"]
