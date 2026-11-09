"""Create an actual-versus-predicted plot for the chronological test set."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd

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
    test["Predicted_AQI"] = model.predict(test[FEATURES])
    sample = test.head(100)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "actual_vs_predicted.png"

    plt.figure(figsize=(12, 5))
    plt.plot(sample["Next_Day_AQI"].to_numpy(), label="Actual AQI")
    plt.plot(sample["Predicted_AQI"].to_numpy(), label="Predicted AQI")
    plt.title(f"AirAware: Actual vs Predicted AQI (test starts {test_start.date()})")
    plt.xlabel("Test observation")
    plt.ylabel("AQI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    mae = ((test["Next_Day_AQI"] - test["Predicted_AQI"]).abs()).mean()
    print(f"Prediction evaluation complete. Test MAE: {mae:.2f} AQI points.")
    print(f"Graph saved to: {output_path}")


if __name__ == "__main__":
    main()
