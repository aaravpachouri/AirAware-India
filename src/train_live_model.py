"""Train and evaluate the separate live AQI forecast model.

Evaluation is chronological and uses training-only imputation. The final
artifact is then refit on all eligible historical rows for deployment. These
city-day metrics do not establish station-level live forecast accuracy.
"""
from __future__ import annotations

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from evaluation_utils import ROOT, chronological_split, load_ml_data


LIVE_FEATURES = ["AQI", "PM2.5", "PM10", "Day", "Month", "DayOfWeek"]
POLLUTANT_FEATURES = ["PM2.5", "PM10"]
MODEL_PATH = ROOT / "models" / "airaware_live_rf.joblib"


def build_live_model() -> Pipeline:
    """Build a compact live-model pipeline with train-fitted pollutant imputation."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "pollutant_imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
                POLLUTANT_FEATURES,
            )
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_ml_data()
    train, test, test_start = chronological_split(df)

    # Evaluate before refitting on the full dataset. The test period is never
    # used to fit the evaluation imputer or the evaluation model.
    evaluation_model = build_live_model()
    evaluation_model.fit(train[LIVE_FEATURES], train["Next_Day_AQI"])
    predictions = evaluation_model.predict(test[LIVE_FEATURES])
    actual = test["Next_Day_AQI"]

    model_mae = mean_absolute_error(actual, predictions)
    model_r2 = r2_score(actual, predictions)
    persistence_mae = mean_absolute_error(actual, test["AQI"])
    three_day_mae = mean_absolute_error(actual, test["AQI_3Day_Avg"])

    print("LIVE FORECAST MODEL — CHRONOLOGICAL EVALUATION")
    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Last training feature date: {train['Date'].max().date()}")
    print(f"First test feature date: {test_start.date()}")
    print(f"Model MAE: {model_mae:.2f} AQI points")
    print(f"Model R²: {model_r2:.3f}")
    print(f"Persistence baseline MAE: {persistence_mae:.2f} AQI points")
    print(f"Three-day mean baseline MAE: {three_day_mae:.2f} AQI points")

    # The deployable artifact is retrained on all eligible historical rows
    # only after the holdout metrics have been calculated.
    deployment_model = build_live_model()
    deployment_model.fit(df[LIVE_FEATURES], df["Next_Day_AQI"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployment_model, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved deployment model: {MODEL_PATH}")
    print(f"Deployment model size: {size_mb:.1f} MiB")
    print(
        "Limitation: this evaluation uses historical city-day observations; "
        "it does not validate forecasts against future live station readings."
    )


if __name__ == "__main__":
    main()
