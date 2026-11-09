"""Shared, reproducible dataset and chronological split helpers for AirAware.

The target is next-calendar-day AQI. Training rows must end before the first
test feature date's prior calendar day to avoid overlapping next-day targets.
Pollutant NaNs are allowed because the model imputes them using training data.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ml_data.csv"

# Keep this exact order in sync with app.py's historical forecast inputs.
FEATURES = [
    "City",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI",
    "Previous_Day_AQI",
    "AQI_3Day_Avg",
    "AQI_Change",
    "Day",
    "Month",
    "DayOfWeek",
]

POLLUTANT_FEATURES = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
NON_IMPUTABLE_FEATURES = [
    "AQI",
    "Previous_Day_AQI",
    "AQI_3Day_Avg",
    "AQI_Change",
    "Day",
    "Month",
    "DayOfWeek",
]


def load_ml_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"ML dataset not found at {DATA_PATH}. Run the data preparation "
            "pipeline first."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"Date", "City", "Next_Day_AQI", *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "The ML dataset is missing required columns: " + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in [*FEATURES, "Next_Day_AQI"]:
        if column != "City":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Pollutant columns may be missing and are imputed inside the training
    # pipeline. Targets, dates, and lag/rolling/calendar features may not.
    required_observations = ["Date", "City", "Next_Day_AQI", *NON_IMPUTABLE_FEATURES]
    df = df.dropna(subset=required_observations)
    df = df.sort_values(["Date", "City"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("The ML dataset contains no valid dated target rows.")
    return df


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by unique dates and keep next-day target windows disjoint."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in the range [0.5, 1.0).")

    dates = pd.DatetimeIndex(df["Date"].dropna().drop_duplicates().sort_values())
    if len(dates) < 4:
        raise ValueError("At least four distinct dates are required for a temporal split.")

    cutoff_index = int(len(dates) * train_fraction)
    cutoff_index = min(max(cutoff_index, 2), len(dates) - 1)
    test_start = pd.Timestamp(dates[cutoff_index])

    # Next_Day_AQI at date D is the observed AQI for D+1. Exclude the
    # immediately preceding feature date from training so no training label
    # lands on the first test feature date.
    train_end_exclusive = test_start - pd.Timedelta(days=1)
    train = df.loc[df["Date"] < train_end_exclusive].copy()
    test = df.loc[df["Date"] >= test_start].copy()

    if train.empty or test.empty:
        raise ValueError(
            "The chronological split produced an empty train or test set. "
            "Check the date coverage in data/ml_data.csv."
        )
    if train["Date"].max() + pd.Timedelta(days=1) >= test["Date"].min():
        raise AssertionError("Training labels may overlap the test feature window.")

    return train, test, test_start
