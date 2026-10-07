import pandas as pd
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/ml_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

split_index = int(len(df) * 0.8)

test = df.iloc[split_index:]

actual = test["Next_Day_AQI"]
baseline_prediction = test["AQI"]

mae = mean_absolute_error(actual, baseline_prediction)

print(f"Naive baseline MAE: {mae:.2f}")