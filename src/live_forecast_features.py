"""Validated feature construction for the live AQI forecasting model."""
from __future__ import annotations

import math

import pandas as pd


LIVE_FEATURES = ["AQI", "PM2.5", "PM10", "Day", "Month", "DayOfWeek"]
OPTIONAL_POLLUTANT_FEATURES = {"PM2.5": "pm25", "PM10": "pm10"}


def _required_non_negative(value, feature: str) -> float:
    """Validate a required numeric model input such as the current AQI."""
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Forecast requires a valid {feature} measurement.") from None
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(
            f"Forecast requires a finite, non-negative {feature} measurement."
        )
    return normalized


def _optional_pollutant(value, feature: str) -> float:
    """Preserve absent pollutant data as NaN for the model's fitted imputer."""
    if value is None or pd.isna(value):
        return float("nan")
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Forecast received an invalid {feature} measurement."
        ) from None
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(
            f"Forecast requires a finite, non-negative {feature} measurement."
        )
    return normalized


def build_live_forecast_features(
    current_aqi: float,
    pollutants: dict,
    timestamp,
) -> pd.DataFrame:
    """Build training-schema inputs; fitted training imputation handles gaps.

    PM2.5 and PM10 are optional because the deployed training pipeline fits a
    median imputer on historical training data for these exact features.
    Missing observations remain NaN here and are never replaced with zero.
    """
    normalized = {
        "AQI": _required_non_negative(current_aqi, "AQI"),
        "PM2.5": _optional_pollutant(pollutants.get("pm25"), "PM2.5"),
        "PM10": _optional_pollutant(pollutants.get("pm10"), "PM10"),
    }

    try:
        observed_at = pd.Timestamp(timestamp)
    except (TypeError, ValueError):
        raise ValueError("Forecast requires a valid measurement timestamp.") from None
    if pd.isna(observed_at):
        raise ValueError("Forecast requires a valid measurement timestamp.")

    return pd.DataFrame(
        [{
            "AQI": normalized["AQI"],
            "PM2.5": normalized["PM2.5"],
            "PM10": normalized["PM10"],
            "Day": observed_at.day,
            "Month": observed_at.month,
            "DayOfWeek": observed_at.dayofweek,
        }],
        columns=LIVE_FEATURES,
    )
