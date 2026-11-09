"""Tests for live forecast input validation and model feature schema."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_forecast_features import LIVE_FEATURES, build_live_forecast_features


def test_build_live_forecast_features_matches_training_schema():
    features = build_live_forecast_features(
        current_aqi=120,
        pollutants={"pm25": 35.5, "pm10": 60.0},
        timestamp="2026-11-10T08:30:00Z",
    )
    assert list(features.columns) == LIVE_FEATURES
    assert features.loc[0, "AQI"] == 120
    assert features.loc[0, "PM2.5"] == 35.5
    assert features.loc[0, "PM10"] == 60.0
    assert features.loc[0, "DayOfWeek"] == pd.Timestamp("2026-11-10").dayofweek


@pytest.mark.parametrize(
    "pollutants",
    [
        {"pm25": None, "pm10": 50},
        {"pm25": 20, "pm10": None},
        {},
    ],
)
def test_missing_live_pollutants_are_not_substituted_with_zero(pollutants):
    with pytest.raises(ValueError, match="missing values are not replaced with zero"):
        build_live_forecast_features(100, pollutants, "2026-11-10T08:30:00Z")


def test_invalid_timestamp_is_rejected():
    with pytest.raises(ValueError, match="valid measurement timestamp"):
        build_live_forecast_features(100, {"pm25": 20, "pm10": 50}, "not-a-date")



def test_negative_live_pollutant_is_rejected():
    with pytest.raises(ValueError, match="finite, non-negative PM2.5"):
        build_live_forecast_features(
            100,
            {"pm25": -1, "pm10": 50},
            "2026-11-10T08:30:00Z",
        )
