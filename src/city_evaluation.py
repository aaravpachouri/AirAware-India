"""Evaluate next-day AQI errors by city using the trained historical model.

Run from the repository root:
    python src/city_evaluation.py

This script deliberately uses the same feature order as src/train_model.py.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ml_data.csv"
MODEL_PATH = ROOT / "models" / "airaware_random_forest.joblib"

FEATURES = [
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
]


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run the data preparation "
            "pipeline first and ensure data/ml_data.csv exists."
        )
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run "
            "python src/train_model.py first."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"Date", "City", "Next_Day_AQI", *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "The ML dataset is missing required columns: " + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "City", "Next_Day_AQI"])
    df = df.sort_values(["Date", "City"]).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("At least two valid rows are required for evaluation.")

    # Keep the evaluation window consistent with the existing 80/20
    # chronological row split used by train_model.py.
    split_index = int(len(df) * 0.8)
    split_index = min(max(split_index, 1), len(df) - 1)
    test = df.iloc[split_index:].copy()

    model = joblib.load(MODEL_PATH)
    test["Prediction"] = model.predict(test[FEATURES])

    results = (
        test.groupby("City", sort=True)
        .apply(
            lambda group: mean_absolute_error(
                group["Next_Day_AQI"], group["Prediction"]
            ),
            include_groups=False,
        )
        .sort_values()
    )

    print(f"Evaluation rows: {len(test)}")
    print("MAE by city (AQI points; lower is better):")
    print(results.to_string())


if __name__ == "__main__":
    main()
