import pandas as pd

df = pd.read_csv("data/cleaned_city_day.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City", "Date"])
df["Previous_Date"] = df.groupby("City")["Date"].shift(1)
df["Previous_Day_AQI"] = df.groupby("City")["AQI"].shift(1)

df.loc[
     (df["Date"] - df["Previous_Date"]) != pd.Timedelta(days=1),
     "Previous_Day_AQI"
] = None

df["AQI_Change"] = df["AQI"] - df["Previous_Day_AQI"]
df["AQI_3Day_Avg"] = (
    df.groupby("City")["AQI"]
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Tomorrow's AQI becomes our prediction target
df["Next_Date"] = df.groupby("City")["Date"].shift(-1)
df["Next_Day_AQI"] = df.groupby("City")["AQI"].shift(-1)

df.loc[
     (df["Next_Date"] - df["Date"]) != pd.Timedelta(days=1),
     "Next_Day_AQI"
] = None    

# Time-based features
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

features = [
    "Date",
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
    "Next_Day_AQI"
]

ml_df = df[features].dropna()

ml_df.to_csv("data/ml_data.csv", index=False)

print("ML dataset created.")
print("Shape:", ml_df.shape)
print("\nColumns:")
print(ml_df.columns.tolist())