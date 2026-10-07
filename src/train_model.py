import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load data
df = pd.read_csv("data/ml_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
df = df.sort_values("Date")

# Features and target
X = df[
    [
        "City",
        "Previous_Day_AQI",
        "AQI_3Day_Avg",
        "AQI_Change",
        "PM2.5",
        "PM10",
        "NO2",
        "CO",
        "SO2",
        "O3",
        "AQI",
        "Day",
        "Month",
        "DayOfWeek",
    ]
]

y = df["Next_Day_AQI"]

# Chronological 80/20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Handle City + numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(handle_unknown="ignore"),
            ["City"],
        )
    ],
    remainder="passthrough",
)

# Random Forest model
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

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training complete.")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# Save model
joblib.dump(model, "models/airaware_random_forest.joblib")

print("\nModel saved to:")
joblib.dump(model, "models/airaware_random_forest_v3.joblib")
print("models/airaware_random_forest_v3.joblib")