import pandas as pd

df = pd.read_csv("data/city_day.csv")

columns = [
    "City",
    "Date",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI"
]

df = df[columns]

# Remove rows where AQI is missing
df = df.dropna(subset=["AQI"])
df = df[df["AQI"].between(0, 500)]

# Fill missing pollution values with median
pollution_columns = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

for column in pollution_columns:
    df[column] = df[column].fillna(df[column].median())

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Sort by city and date
df = df.sort_values(["City", "Date"])

# Save cleaned dataset
df.to_csv("data/cleaned_city_day.csv", index=False)

print("Cleaned dataset saved.")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())