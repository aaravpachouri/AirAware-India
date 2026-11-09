# AirAware India — Dataset Card

## Dataset identity

- **Dataset:** Air Quality Data in India (2015–2020), city-day observations
- **Prediction unit:** one city on one calendar day
- **Primary target:** AQI observed on the next calendar day
- **Historical period:** 2015 through 2020 (source coverage is not complete for every city/date)
- **Raw file:** city_day.csv
- **Original distribution:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- **Pinned reproducibility mirror:** https://github.com/govindlabhala-cell/India-Air-Quality-EDA
- **Dataset catalog / license metadata:** https://portal.transport-data.org/@tumi/urban-air-pollution-india-2015-2020
- **License reported by TUMI catalog:** CC0 1.0
- **Pinned SHA-256:** 0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847

The GitHub mirror is used as a pinned delivery location for reproducible builds. It is not represented as the original data publisher. The downloader fails if the downloaded bytes do not match the pinned checksum.

## Fields used by the historical model

| Field | Meaning | Use |
|---|---|---|
| City | City name attached to the historical record | Categorical predictor in the city-aware model |
| Date | Observation date | Chronological split and feature cutoff |
| PM2.5 | Fine particulate concentration | Predictor; missing values imputed inside the training pipeline |
| PM10 | Coarse particulate concentration | Predictor; missing values imputed inside the training pipeline |
| NO2 | Nitrogen dioxide concentration | Predictor; missing values imputed inside the training pipeline |
| CO | Carbon monoxide concentration | Predictor; missing values imputed inside the training pipeline |
| SO2 | Sulfur dioxide concentration | Predictor; missing values imputed inside the training pipeline |
| O3 | Ozone concentration | Predictor; missing values imputed inside the training pipeline |
| AQI | Historical AQI observation | Predictor and persistence baseline |
| Next_Day_AQI | AQI observed on the next calendar day | Regression target |

Engineered predictors include previous-day AQI, the mean AQI of the last three consecutive calendar days, AQI change, day, month, and day of week.

## Data preparation

1. Verify the raw file checksum.
2. Parse dates and numeric values.
3. Remove rows without a valid city, date, or AQI target.
4. Keep AQI observations within the source model's supported 0–500 range.
5. Retain missing pollutant values rather than computing imputation statistics over the full dataset.
6. Reject duplicate city/date rows because they make lags and rolling windows ambiguous.
7. Compute previous-day AQI only when the preceding observation is exactly one calendar day earlier.
8. Compute the three-day mean only when all three calendar dates are consecutive.
9. Create the target only when the next observation is exactly one calendar day later.
10. Fit pollutant imputation medians on training data only.

## Known limitations and risks

- **City-day aggregation:** the historical dataset is not station-level data. A city-level model cannot be assumed to predict every monitoring station accurately.
- **Missingness:** pollutant data is incomplete and missingness may not be random.
- **Measurement provenance:** the dataset is a historical compilation distributed through third-party sources. Check source metadata before making regulatory claims.
- **Temporal coverage:** not all cities have observations on all dates; rows are not independent samples.
- **Distribution shift:** unseen cities and live station measurements can differ from the training distribution.
- **Target limitations:** historical AQI is used as supplied by the source. The live dashboard separately estimates a CPCB-style AQI from recent OpenAQ measurements; those estimates are not official CPCB values.
- **Use limitations:** not a medical, legal, regulatory, or emergency-warning system.

## Appropriate uses

- Educational environmental-data research
- Reproducible model evaluation
- Exploratory analysis of historical city-level air-quality patterns
- Testing short-horizon forecasting methods and their limitations

## Inappropriate uses

- Claiming official government AQI values from the live estimate
- Health diagnoses or individualized medical recommendations
- Regulatory compliance decisions
- Claiming reliable forecasts for a city or station without validating that deployment setting

## Citation

Air Quality Data in India (2015–2020), distributed through Kaggle by Rohan Rao; dataset metadata and CC0 1.0 license information are listed by the Transformative Urban Mobility Initiative data portal. The specific mirror URL and SHA-256 checksum used by this repository are documented above.
