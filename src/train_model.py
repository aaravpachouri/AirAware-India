"""Train and evaluate AirAware's historical next-day AQI model.

Run from the repository root:
    python src/train_model.py
"""
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluation_utils import FEATURES, ROOT, chronological_split, load_ml_data


def main() -> None:
    df = load_ml_data()
    train, test, test_start = chronological_split(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("city", OneHotEncoder(handle_unknown="ignore"), ["City"]),
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(train[FEATURES], train["Next_Day_AQI"])
    predictions = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["Next_Day_AQI"], predictions)
    r2 = r2_score(test["Next_Day_AQI"], predictions)

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Keep both historical filenames because other project scripts may use them.
    joblib.dump(model, model_dir / "airaware_random_forest.joblib")
    joblib.dump(model, model_dir / "airaware_random_forest_v3.joblib")

    print("Model training complete.")
    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")
    print(f"First test feature date: {test_start.date()}")
    print(f"Last training feature date: {train['Date'].max().date()}")
    print(f"MAE: {mae:.2f} AQI points")
    print(f"R²: {r2:.3f}")
    print(f"Models saved in: {model_dir}")


if __name__ == "__main__":
    main()
