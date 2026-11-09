"""Compare simple forecasting baselines on the shared chronological test set."""
from sklearn.metrics import mean_absolute_error

from evaluation_utils import chronological_split, load_ml_data


def main() -> None:
    df = load_ml_data()
    _, test, test_start = chronological_split(df)
    actual = test["Next_Day_AQI"]

    baselines = {
        "Persistence (tomorrow = today)": test["AQI"],
        "Three-day mean (last 3 consecutive days)": test["AQI_3Day_Avg"],
    }

    print(f"First test feature date: {test_start.date()}")
    print(f"Test rows: {len(test)}")
    print("Baseline MAE (AQI points; lower is better):")
    for name, prediction in baselines.items():
        mae = mean_absolute_error(actual, prediction)
        print(f"{name}: {mae:.2f}")


if __name__ == "__main__":
    main()
