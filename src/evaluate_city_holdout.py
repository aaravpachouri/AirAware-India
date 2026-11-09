"""Compare city-aware and city-agnostic models on held-out cities.

This answers a different question from chronological evaluation: transfer to
cities excluded from fitting. The result is a fixed deterministic holdout,
not evidence that the model generalizes to every Indian city.
"""
from sklearn.metrics import mean_absolute_error, r2_score

from evaluation_utils import FEATURES, city_holdout_split, load_ml_data
from train_model import build_model


def report_model(name, model, train, test):
    model.fit(train[FEATURES], train["Next_Day_AQI"])
    predictions = model.predict(test[FEATURES])
    actual = test["Next_Day_AQI"]

    print(f"\n{name}")
    print(f"MAE: {mean_absolute_error(actual, predictions):.2f} AQI points")
    print(f"R²: {r2_score(actual, predictions):.3f}")
    print("MAE by held-out city:")
    for city, indices in test.groupby("City", sort=True).groups.items():
        group_actual = actual.loc[indices]
        group_predictions = predictions[list(test.index).index(indices[0]):list(test.index).index(indices[0]) + len(indices)]
        city_mae = mean_absolute_error(group_actual, group_predictions)
        baseline_mae = mean_absolute_error(group_actual, test.loc[indices, "AQI"])
        print(
            f"{city}: model={city_mae:.2f}, "
            f"persistence={baseline_mae:.2f}, n={len(indices)}"
        )


def main() -> None:
    df = load_ml_data()
    train, test, held_out_cities = city_holdout_split(df)

    print("CITY-HELD-OUT GENERALIZATION EVALUATION")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Training cities: {train['City'].nunique()}")
    print(f"Held-out cities: {', '.join(held_out_cities)}")
    print(f"Persistence baseline MAE: {mean_absolute_error(test['Next_Day_AQI'], test['AQI']):.2f}")
    print(f"Three-day mean baseline MAE: {mean_absolute_error(test['Next_Day_AQI'], test['AQI_3Day_Avg']):.2f}")

    report_model("City-aware Random Forest", build_model(include_city=True), train, test)
    report_model("City-agnostic Random Forest", build_model(include_city=False), train, test)


if __name__ == "__main__":
    main()
