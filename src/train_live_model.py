import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/ml_data.csv"
MODEL_PATH = "models/airaware_live_rf.joblib"


df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

features = [
    "AQI",
    "PM2.5",
    "PM10",
    "Day",
    "Month",
    "DayOfWeek",
]

target = "Next_Day_AQI"

df = df.dropna(
    subset=features + [target]
)

X = df[features]
y = df[target]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== AIR AWARE LIVE MODEL ==========\n")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print(f"\nMAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

joblib.dump(
    model,
    MODEL_PATH
)

print("\nSaved:")
print(MODEL_PATH)