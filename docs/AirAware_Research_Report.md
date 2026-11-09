# AirAware India - Research Report

## Abstract

AirAware India is a research-oriented machine-learning system for short-term air-quality forecasting in India. The project combines historical city-level air-quality observations with a public-facing monitoring and forecasting dashboard.

The core research task is next-day AQI prediction using pollutant, temporal, and recent AQI features. A Random Forest regression model achieved a test MAE of **16.41 AQI points** and an R² of **0.890** on the held-out historical test set. A persistence baseline, defined as predicting tomorrow's AQI as today's AQI, achieved an MAE of **18.81 AQI points** on the corresponding evaluation data.

The project also includes an OpenAQ-based monitoring pipeline that discovers Indian monitoring stations, retrieves recent pollutant observations, calculates an **estimated CPCB-style AQI**, and combines the current state with the forecasting model to produce an understandable tomorrow outlook.

---

## 1. Motivation

Air-quality information is often presented as raw pollutant concentrations or technical indices. Although useful for research, these measurements can be difficult for non-technical users to interpret.

AirAware explores a practical question:

> **Can a lightweight machine-learning model provide useful short-term air-quality forecasts at local monitoring locations using pollutant and temporal features?**

The project was designed around two goals:

1. Build a technically grounded forecasting system.
2. Translate the output into understandable environmental information.

---

## 2. Research Question

> Can lightweight machine-learning models provide useful short-term air-quality forecasts at local monitoring locations using pollutant and temporal features?

The primary prediction target is:

```text
Next_Day_AQI
```

The task is therefore a supervised regression problem.

---

## 3. Historical Dataset

The forecasting pipeline was developed using the **Air Quality Data in India (2015-2020)** city-level dataset.

The original data contains city, date, pollutant measurements, AQI, and AQI category information.

The project retained the main variables required for exploratory analysis and forecasting:

- City
- Date
- PM2.5
- PM10
- NO2
- CO
- SO2
- O3
- AQI

### Data preparation

The cleaning pipeline:

1. Removed observations without a usable AQI target.
2. Restricted AQI to the 0-500 range for the research dataset.
3. Retained missing pollutant values for model-time imputation fitted only on training rows.
4. Converted dates to datetime values.
5. Sorted observations chronologically by city and date.

The cleaned dataset contains **24,307 observations**.

---

## 4. Feature Engineering

The forecasting dataset was constructed chronologically for each city.

Additional features were created to represent recent air-quality conditions:

```text
Previous_Day_AQI
AQI_3Day_Avg
AQI_Change
Day
Month
DayOfWeek
```

A next-day date check was also used so that the target represented the AQI for the actual following calendar day rather than simply the next available row.

The resulting machine-learning dataset contains **22,468 rows before final evaluation filtering**.

---

## 5. Model

### Algorithm

**Random Forest Regressor**

The final historical model uses:

- 150 estimators
- fixed random seed
- parallel training
- one-hot encoding for city
- numerical pollutant and temporal features

### Feature set

```text
City
PM2.5
PM10
NO2
CO
SO2
O3
AQI
Previous_Day_AQI
AQI_3Day_Avg
AQI_Change
Day
Month
DayOfWeek
```

### Validation strategy

The data was split chronologically using an **80/20 train-test split**.

This prevents future observations from being used to train the model before earlier observations are evaluated.

---

## 6. Results

GitHub Actions rebuilt the historical dataset from a checksum-verified raw source and retrained the Random Forest using a date-based split with training-only pollutant imputation.

| Evaluation detail | Result |
|---|---:|
| Training rows | 14,232 |
| Held-out test rows | 8,220 |
| Last training feature date | 2019-05-24 |
| First test feature date | 2019-05-26 |
| Test MAE | 16.41 AQI points |
| Test R² | 0.890 |
| Persistence baseline MAE | 18.81 AQI points |
| Three-day mean baseline MAE | 22.70 AQI points |

The fresh model's MAE was approximately 12.8% lower than the persistence baseline, measured on the same held-out rows. Performance varies by city, so the aggregate result should not be interpreted as uniform accuracy across India.

### Separate city-held-out experiment

A deterministic group split held out six complete cities (4,508 observations) that were not used for fitting. The city-aware Random Forest achieved MAE **28.25 AQI points** (R² 0.545); a city-agnostic Random Forest achieved MAE **24.64** (R² 0.703); persistence achieved MAE **22.19**; and the three-day mean achieved MAE **24.64**. Neither Random Forest beat persistence on this split. This shows that the model's chronological result does not establish generalization to unseen cities. Per-city breakdowns are being revalidated separately before being reported.

### Training and test protocol

- The prediction target is AQI on the next calendar day.
- Rows are sorted and split by unique dates, not by randomly shuffling individual observations.
- A one-day gap is left between the last training feature date and the first test feature date to prevent the final training target from overlapping the test feature window.
- Missing pollutant values are retained through dataset preparation and imputed by a median imputer fitted only on the training rows.
- The raw source CSV is downloaded from a public mirror and verified against SHA-256 `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847`. The raw CSV is not committed to the repository.

### Legacy artifact caution

The previously checked-in model artifact scored MAE 11.39 AQI points and R² 0.938 when evaluated on a later-date holdout. Its original training cutoff and split method are undocumented, so it may have been trained on future dates relative to the holdout. We do not accept that stronger score as reliable out-of-sample evidence. The leakage-controlled fresh retrain is the reported result.

R² describes variation explained relative to a constant-mean predictor; it is not a percentage accuracy score. These results concern historical city-level data and should not be interpreted as forecasts of current air quality.

---

## 7. Feature Importance

The leakage-controlled training run identified current AQI as the dominant predictor. The values below were emitted by the same successful GitHub Actions run as the evaluation metrics:

| Feature | Importance |
|---|---:|
| AQI | 0.803441 |
| PM2.5 | 0.075267 |
| PM10 | 0.017466 |
| CO | 0.017024 |
| AQI_3Day_Avg | 0.014914 |
| NO2 | 0.010235 |
| O3 | 0.009664 |
| AQI_Change | 0.009662 |
| SO2 | 0.009638 |
| Previous_Day_AQI | 0.007309 |

This is consistent with the short forecasting horizon: current AQI carries substantial information about the following day's state. Feature importance is descriptive, not causal.

---

## 8. Live Monitoring Pipeline

AirAware also integrates recent monitoring observations through **OpenAQ**.

The live workflow is:

```text
OpenAQ
   ↓
Discover Indian monitoring stations
   ↓
Select station
   ↓
Fetch latest measurements
   ↓
Map sensor IDs to pollutants
   ↓
Normalize units
   ↓
Calculate pollutant sub-indices
   ↓
Estimate CPCB-style AQI
   ↓
Generate AirAware forecast
   ↓
Present understandable insight
```

The dashboard can expose recent station measurements for:

- PM2.5
- PM10
- NO2
- SO2
- CO
- O3

---

## 9. AQI Method

AirAware calculates an **estimated CPCB-style AQI** from available pollutant concentrations using the configured pollutant breakpoints and interpolation logic.

The highest available pollutant sub-index determines the displayed estimated AQI.

This is deliberately labelled as an estimate.

### Why it is not official CPCB AQI

Official AQI calculations use prescribed averaging periods and sufficient observations. A latest station measurement retrieved from a live API does not necessarily satisfy those requirements.

Therefore:

> **AirAware's live AQI should not be interpreted as an official CPCB AQI value.**

This distinction is a core part of the project's research methodology.

---

## 10. Public-Facing Information Layer

The forecasting result is translated into simple language rather than exposing only a numerical prediction.

The dashboard communicates:

- current AQI state
- tomorrow's forecast
- expected direction of change
- pollutant measurements
- pollutant contribution estimates
- plain-language environmental guidance

The guidance is intentionally non-medical.

---

## 11. Dashboard

The application is implemented with Streamlit.

### Main pages

#### Latest Air

Displays:

- monitoring station
- current estimated AQI
- AQI category
- pollutant measurements
- tomorrow forecast
- AQI movement
- pollutant sub-indices
- environmental insight

#### Historical Research

Allows exploration of:

- city
- date
- today's AQI
- predicted next-day AQI
- actual next-day AQI where available
- prediction error
- recent AQI trend

#### Model Performance

Documents:

- forecasting algorithm
- MAE
- R²
- model inputs
- limitations

---

## 12. Engineering Architecture

```text
                    ┌─────────────────────┐
                    │ Historical Dataset  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cleaning + EDA      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest Model  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation     │
                    └──────────┬──────────┘
                               │
                               │
OpenAQ ──► Station ──► Pollutants ──► AQI ──► Forecast
                               │              │
                               └──────┬───────┘
                                      ▼
                             Streamlit Dashboard
```

---

## 13. Research Visualizations

The project includes:

- `reports/aqi_distribution.png`
- `reports/average_aqi_by_city.png`
- `reports/pm25_vs_aqi.png`
- `reports/actual_vs_predicted.png`

These visualizations support the exploratory analysis and model evaluation.

---

## 14. Limitations

AirAware is a research prototype rather than an official air-quality service.

### Data limitations

The historical model was trained on city-level historical observations. Live data comes from monitoring stations and can have different spatial and temporal characteristics.

### AQI limitations

The live estimate does not necessarily reproduce the official averaging procedure used for government AQI reporting.

### Forecast limitations

The persistence baseline is strong, showing that short-horizon air-quality forecasting is inherently challenging.

### Distribution limitations

Model performance varies between cities and pollution regimes.

### Operational limitations

Live station measurements can be delayed or unavailable.

---

## 15. Future Research

Possible extensions include:

- stronger temporal validation
- city-held-out evaluation
- station-specific models
- meteorological features
- probabilistic forecasts
- prediction intervals
- multi-day forecasting
- improved missing-data handling
- calibration of forecast uncertainty
- broader public-impact evaluation

---

## 16. Reproducibility

The repository contains the main data-processing, training, evaluation, and application code.

Core scripts include:

```text
src/download_source_data.py
src/clean_data.py
src/eda.py
src/prepare_ml_data.py
src/train_model.py
src/train_live_model.py
src/evaluate_saved_model.py
src/evaluate_baseline.py
src/evaluate_predictions.py
src/feature_importance.py
src/city_evaluation.py
src/live_air.py
src/live_forecast.py
```

Rebuild the research model reproducibly with:

```bash
pip install -r requirements-dev.txt
python src/download_source_data.py
python src/clean_data.py
python src/prepare_ml_data.py
python src/train_model.py
python src/evaluate_baseline.py
python src/evaluate_predictions.py
python src/city_evaluation.py
python src/feature_importance.py
```

Run the interactive dashboard with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live monitoring requires an OpenAQ API key configured through the local environment or hosted deployment secrets.

---

## 17. Conclusion

AirAware India demonstrates a complete lightweight environmental ML workflow:

```text
Data
  ↓
Cleaning
  ↓
Exploration
  ↓
Feature Engineering
  ↓
Forecasting
  ↓
Baseline Evaluation
  ↓
Live Data Integration
  ↓
Public-Facing Application
```

The project shows that a relatively lightweight model can provide useful short-term AQI forecasts while also exposing the practical challenges of real-world environmental data, live monitoring, and public interpretation.

The central contribution is not only the prediction model, but the integration of **data engineering, machine learning evaluation, live environmental data, and understandable communication into one reproducible system**.

---

## Project Status

**Public research prototype**

GitHub:
https://github.com/aaravpachouri/AirAware-India

Application:
Deployed through Streamlit Community Cloud.

---

*AirAware India was developed as an independent research and engineering project.*
