"""Create next-calendar-day AQI features while retaining pollutant NaNs.

Rows missing target or essential lag/rolling features are removed. Pollutant
missing values remain and are imputed by the model's training-fitted pipeline.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "cleaned_city_day.csv"
OUTPUT_PATH = DATA_DIR / "ml_data.csv"

FEATURES = [
    "Date",
    "City",
    "Previous_Day_AQI",
    "AQI_3Day_Avg",
    "AQI_Change",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI",
    "Day",
    "Month",
    "DayOfWeek",
    "Next_Day_AQI",
]

REQUIRED_NON_IMPUTABLE = [
    "Date",
    "City",
    "Previous_Day_AQI",
    "AQI_3Day_Avg",
    "AQI_Change",
    "AQI",
    "Day",
    "Month",
    "DayOfWeek",
    "Next_Day_AQI",
]


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {INPUT_PATH}. "
            "Run python src/clean_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    df["Previous_Date"] = df.groupby("City")["Date"].shift(1)
    df["Previous_Day_AQI"] = df.groupby("City")["AQI"].shift(1)
    df.loc[
        (df["Date"] - df["Previous_Date"]) != pd.Timedelta(days=1),
        "Previous_Day_AQI",
    ] = None

    df["AQI_Change"] = df["AQI"] - df["Previous_Day_AQI"]
    df["AQI_3Day_Avg"] = (
        df.groupby("City")["AQI"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Next_Date"] = df.groupby("City")["Date"].shift(-1)
    df["Next_Day_AQI"] = df.groupby("City")["AQI"].shift(-1)
    df.loc[
        (df["Next_Date"] - df["Date"]) != pd.Timedelta(days=1),
        "Next_Day_AQI",
    ] = None

    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    pollutant_columns = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    for column in pollutant_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    ml_df = df[FEATURES].dropna(subset=REQUIRED_NON_IMPUTABLE).copy()
    ml_df = ml_df.sort_values(["Date", "City"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(OUTPUT_PATH, index=False)

    print(f"ML dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {ml_df.shape}")
    print("Missing pollutant values to be imputed within each training split:")
    print(ml_df[pollutant_columns].isna().sum().to_string())


if __name__ == "__main__":
    main()
