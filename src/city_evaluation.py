"""Evaluate next-day AQI errors by city on the shared chronological test set."""
import joblib
from sklearn.metrics import mean_absolute_error

from evaluation_utils import FEATURES, ROOT, chronological_split, load_ml_data


def main() -> None:
    df = load_ml_data()
    _, test, test_start = chronological_split(df)
    model_path = ROOT / "models" / "airaware_random_forest_v3.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run python src/train_model.py first."
        )

    model = joblib.load(model_path)
    test = test.copy()
    test["Prediction"] = model.predict(test[FEATURES])

    rows = []
    for city, group in test.groupby("City", sort=True):
        rows.append(
            {
                "City": city,
                "Test rows": len(group),
                "MAE (AQI points)": mean_absolute_error(
                    group["Next_Day_AQI"], group["Prediction"]
                ),
            }
        )

    results = sorted(rows, key=lambda row: row["MAE (AQI points)"])
    print(f"First test feature date: {test_start.date()}")
    print("MAE by city (AQI points; lower is better):")
    for row in results:
        print(
            f"{row['City']}: MAE={row['MAE (AQI points)']:.2f} "
            f"(n={row['Test rows']})"
        )


if __name__ == "__main__":
    main()
