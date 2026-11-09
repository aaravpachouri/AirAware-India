"""Evaluate generalization to cities excluded from model training.

This answers a different question from the chronological holdout: can the model
transfer to city groups it did not see during fitting?
"""
from sklearn.metrics import mean_absolute_error, r2_score

from evaluation_utils import (
    FEATURES,
    city_holdout_split,
    load_ml_data,
)
from train_model import build_model


def main() -> None:
    df = load_ml_data()
    train, test, held_out_cities = city_holdout_split(df)

    model = build_model()
    model.fit(train[FEATURES], train["Next_Day_AQI"])
    test = test.copy()
    test["Prediction"] = model.predict(test[FEATURES])

    actual = test["Next_Day_AQI"]
    persistence_mae = mean_absolute_error(actual, test["AQI"])
    three_day_mae = mean_absolute_error(actual, test["AQI_3Day_Avg"])
    model_mae = mean_absolute_error(actual, test["Prediction"])
    model_r2 = r2_score(actual, test["Prediction"])

    print("CITY-HELD-OUT GENERALIZATION EVALUATION")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Training cities: {train['City'].nunique()}")
    print(f"Held-out cities: {', '.join(held_out_cities)}")
    print(f"Model MAE: {model_mae:.2f} AQI points")
    print(f"Model R²: {model_r2:.3f}")
    print(f"Persistence baseline MAE: {persistence_mae:.2f} AQI points")
    print(f"Three-day mean baseline MAE: {three_day_mae:.2f} AQI points")
    print("MAE by held-out city:")
    for city, group in test.groupby("City", sort=True):
        city_model_mae = mean_absolute_error(group["Next_Day_AQI"], group["Prediction"])
        city_baseline_mae = mean_absolute_error(group["Next_Day_AQI"], group["AQI"])
        print(
            f"{city}: model={city_model_mae:.2f}, "
            f"persistence={city_baseline_mae:.2f}, n={len(group)}"
        )


if __name__ == "__main__":
    main()
