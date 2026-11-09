"""Compare Random Forest and Ridge models on temporal and city-held-out splits.

All models are scored on identical rows within each split. Predictions are
clipped to the valid AQI range before metrics are calculated. This script is a
research benchmark and does not change the deployed model artifact.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation_utils import (
    FEATURES,
    chronological_split,
    city_holdout_split,
    load_ml_data,
)
from train_model import build_model


NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "City"]


def build_ridge_model(include_city: bool = True) -> Pipeline:
    """Build a scaled Ridge regression pipeline with training-fold imputation."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    transformers = [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ]
    if include_city:
        transformers.insert(
            0,
            ("city", OneHotEncoder(handle_unknown="ignore"), ["City"]),
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def evaluate_split(split_name, train, test, models):
    print(f"\n=== {split_name} ===")
    print(f"Training rows: {len(train)} | Test rows: {len(test)}")
    if split_name == "City-held-out":
        print(f"Held-out cities: {', '.join(sorted(test['City'].unique()))}")

    actual = test["Next_Day_AQI"]
    print(f"Persistence MAE: {mean_absolute_error(actual, test['AQI']):.2f}")
    print(f"Three-day mean MAE: {mean_absolute_error(actual, test['AQI_3Day_Avg']):.2f}")

    for name, model in models:
        model.fit(train[FEATURES], train["Next_Day_AQI"])
        predictions = np.clip(model.predict(test[FEATURES]), 0, 500)
        print(
            f"{name}: MAE={mean_absolute_error(actual, predictions):.2f}, "
            f"R²={r2_score(actual, predictions):.3f}"
        )
        if split_name == "City-held-out":
            scored = test.copy()
            scored["Prediction"] = predictions
            for city, group in scored.assign(
                Prediction=scored["Prediction"].to_numpy()
            ).groupby("City", sort=True):
                city_mae = mean_absolute_error(group["Next_Day_AQI"], group["Prediction"])
                print(f"  {city}: MAE={city_mae:.2f}, n={len(group)}")


def main() -> None:
    df = load_ml_data()
    temporal_train, temporal_test, temporal_start = chronological_split(df)
    city_train, city_test, held_out = city_holdout_split(df)

    print("MODEL-FAMILY BENCHMARK")
    print(f"Temporal test starts: {temporal_start.date()}")
    models = [
        ("Random Forest (city-aware)", build_model(include_city=True)),
        ("Random Forest (city-agnostic)", build_model(include_city=False)),
        ("Ridge (city-aware)", build_ridge_model(include_city=True)),
        ("Ridge (city-agnostic)", build_ridge_model(include_city=False)),
    ]
    evaluate_split("Chronological", temporal_train, temporal_test, models)
    evaluate_split("City-held-out", city_train, city_test, models)


if __name__ == "__main__":
    main()
