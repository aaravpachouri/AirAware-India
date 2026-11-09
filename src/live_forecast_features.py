"""Validated feature construction for the live AQI forecasting model."""
from __future__ import annotations

import math

import pandas as pd


LIVE_FEATURES = ["AQI", "PM2.5", "PM10", "Day", "Month", "DayOfWeek"]


def build_live_forecast_features(
    current_aqi: float,
    pollutants: dict,
    timestamp,
) -> pd.DataFrame:
    """Build the exact live-model schema without treating missing data as zero."""
    values = {"AQI": current_aqi, "PM2.5": pollutants.get("pm25"), "PM10": pollutants.get("pm10")}
    normalized = {}
    for feature, raw_value in values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Forecast requires a valid {feature} measurement; "
                "missing values are not replaced with zero."
            ) from None
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Forecast requires a finite, non-negative {feature} measurement."
            )
        normalized[feature] = value

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
