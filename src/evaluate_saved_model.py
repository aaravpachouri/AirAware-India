"""Evaluate the checked-in historical model artifact without overwriting it.

Run from the repository root:
    python src/evaluate_saved_model.py
"""
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

from evaluation_utils import FEATURES, ROOT, chronological_split, load_ml_data


def main() -> None:
    model_path = ROOT / "models" / "airaware_random_forest_v3.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Ensure Git LFS objects "
            "were downloaded when checking out the repository."
        )

    df = load_ml_data()
    train, test, test_start = chronological_split(df)
    model = joblib.load(model_path)
    predictions = model.predict(test[FEATURES])

    actual = test["Next_Day_AQI"]
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    baseline_mae = mean_absolute_error(actual, test["AQI"])

    print("CHECKED-IN MODEL ARTIFACT EVALUATION")
    print(f"Artifact: {model_path.name}")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Last training feature date: {train['Date'].max().date()}")
    print(f"First test feature date: {test_start.date()}")
    print(f"Saved-model MAE: {mae:.2f} AQI points")
    print(f"Saved-model R²: {r2:.3f}")
    print(f"Persistence baseline MAE: {baseline_mae:.2f} AQI points")
    print(
        "Provenance: compare this artifact score with the raw-data rebuild "
        "and training-only imputation run in GitHub Actions."
    )


if __name__ == "__main__":
    main()
