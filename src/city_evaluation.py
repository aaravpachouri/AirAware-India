import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/ml_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

split_index = int(len(df) * 0.8)
test = df.iloc[split_index:].copy()

features = [
    "City",
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

model = joblib.load("models/airaware_random_forest.joblib")

test["Prediction"] = model.predict(test[features])

results = (
    test.groupby("City")
    .apply(lambda x: mean_absolute_error(x["Next_Day_AQI"], x["Prediction"]))
    .sort_values()
)

print("MAE by city:")
print(results)