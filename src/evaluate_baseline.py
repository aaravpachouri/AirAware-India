"""Compare AirAware with the persistence baseline on the shared test window."""
from sklearn.metrics import mean_absolute_error

from evaluation_utils import chronological_split, load_ml_data


def main() -> None:
    df = load_ml_data()
    _, test, test_start = chronological_split(df)
    actual = test["Next_Day_AQI"]
    baseline_prediction = test["AQI"]
    mae = mean_absolute_error(actual, baseline_prediction)
    print(f"First test feature date: {test_start.date()}")
    print(f"Test rows: {len(test)}")
    print(f"Persistence baseline MAE: {mae:.2f} AQI points")


if __name__ == "__main__":
    main()
