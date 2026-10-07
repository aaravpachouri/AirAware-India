import pandas as pd
import joblib

model = joblib.load("models/airaware_random_forest_v3.joblib")

preprocessor = model.named_steps["preprocessor"]
forest = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

importance = pd.Series(
    forest.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("Top 10 features:")
print(importance.head(10))