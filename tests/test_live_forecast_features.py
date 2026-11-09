"""Tests for live forecast input validation and model feature schema."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

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
    ("pollutants", "missing_feature"),
    [
        ({"pm25": None, "pm10": 50}, "PM2.5"),
        ({"pm25": 20, "pm10": None}, "PM10"),
        ({}, "PM2.5"),
    ],
)
def test_missing_pollutants_remain_missing_for_fitted_model_imputer(
    pollutants, missing_feature
):
    features = build_live_forecast_features(
        100, pollutants, "2026-11-10T08:30:00Z"
    )
    assert np.isnan(features.loc[0, missing_feature])


def test_missing_pollutant_is_imputed_by_training_style_pipeline_not_zero():
    training = pd.DataFrame(
        {
            "PM2.5": [10.0, 30.0, 50.0],
            "PM10": [20.0, 40.0, 60.0],
        }
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(training)
    features = build_live_forecast_features(
        100, {"pm25": 20.0, "pm10": None}, "2026-11-10T08:30:00Z"
    )
    transformed = imputer.transform(features[["PM2.5", "PM10"]])
    assert transformed[0, 0] == 20.0
    assert transformed[0, 1] == 40.0
    assert transformed[0, 1] != 0.0


def test_missing_current_aqi_is_rejected():
    with pytest.raises(ValueError, match="valid AQI"):
        build_live_forecast_features(
            None, {"pm25": 20, "pm10": 50}, "2026-11-10T08:30:00Z"
        )


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
