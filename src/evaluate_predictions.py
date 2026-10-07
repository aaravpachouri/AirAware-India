import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/ml_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

features = [
    "City",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI",
    "Previous_Day_AQI",
    "AQI_3Day_Avg",
    "AQI_Change",
    "Day",
    "Month",
    "DayOfWeek",
]

split_index = int(len(df) * 0.8)
test = df.iloc[split_index:].copy()

model = joblib.load("models/airaware_random_forest_v3.joblib")

test["Predicted_AQI"] = model.predict(test[features])

# Take the first 100 test observations for visualization
sample = test.head(100)

plt.figure(figsize=(12, 5))

plt.plot(
    sample["Next_Day_AQI"].values,
    label="Actual AQI"
)

plt.plot(
    sample["Predicted_AQI"].values,
    label="Predicted AQI"
)

plt.title("AirAware: Actual vs Predicted AQI")
plt.xlabel("Test Observation")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()

plt.savefig("reports/actual_vs_predicted.png")
plt.close()

print("Prediction evaluation complete.")
print("Graph saved to reports/actual_vs_predicted.png")