"""Clean raw city-day observations without full-dataset imputation.

Pollutant missing values are intentionally retained here. They are imputed by
the model pipeline using medians fitted on the training period only, avoiding
the previous preprocessing leakage from calculating medians before the split.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "city_day.csv"
OUTPUT_PATH = DATA_DIR / "cleaned_city_day.csv"

COLUMNS = [
    "City",
    "Date",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI",
]


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {INPUT_PATH}. "
            "Run python src/download_source_data.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    missing_columns = sorted(set(COLUMNS).difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
    df = df.dropna(subset=["City", "Date", "AQI"])
    df = df[df["AQI"].between(0, 500)].copy()

    pollutant_columns = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
    for column in pollutant_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("Missing pollutant values retained for train-only imputation:")
    print(df[pollutant_columns].isna().sum().to_string())


if __name__ == "__main__":
    main()
