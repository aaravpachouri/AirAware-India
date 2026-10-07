import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_city_day.csv")

# 1. AQI distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["AQI"], bins=40, kde=True)
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.savefig("reports/aqi_distribution.png")
plt.close()

# 2. Average AQI by city
city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
city_aqi.plot(kind="bar")
plt.title("Average AQI by City")
plt.xlabel("City")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/average_aqi_by_city.png")
plt.close()

# 3. PM2.5 vs AQI
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="PM2.5", y="AQI", alpha=0.3)
plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("reports/pm25_vs_aqi.png")
plt.close()

print("EDA complete.")
print("\nHighest average AQI cities:")
print(city_aqi.head(5))

print("\nLowest average AQI cities:")
print(city_aqi.tail(5))