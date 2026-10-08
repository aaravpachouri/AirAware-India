# AirAware India - Research Report

## Abstract

AirAware India is a research-oriented machine-learning system for short-term air-quality forecasting in India. The project combines historical city-level air-quality observations with a public-facing monitoring and forecasting dashboard.

The core research task is next-day AQI prediction using pollutant, temporal, and recent AQI features. A Random Forest regression model achieved a test MAE of **16.20 AQI points** and an R² of **0.883** on the held-out historical test set. A persistence baseline, defined as predicting tomorrow's AQI as today's AQI, achieved an MAE of **18.39 AQI points** on the corresponding evaluation data.

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
3. Filled missing pollutant values for the exploration stage.
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

The resulting machine-learning dataset contains **22,981 observations**.

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

| Evaluation | Result |
|---|---:|
| Test MAE | **16.20 AQI points** |
| Test R² | **0.883** |
| Persistence baseline MAE | **18.39 AQI points** |

### Interpretation

An MAE of 16.20 means that, on average, the model's prediction differs from the observed next-day AQI by about 16 AQI points on the held-out test set.

An R² of 0.883 indicates that the model explains a large share of the variation in the held-out target values.

**R² must not be described as 88.3% prediction accuracy.**

The persistence baseline is important because short-term AQI forecasting can be difficult to improve substantially over simply carrying today's AQI forward. The model therefore needs to be judged against that baseline rather than against accuracy alone.

---

## 7. Feature Importance

The final model showed that current AQI was the dominant predictor.

The leading feature importances were approximately:

| Feature | Importance |
|---|---:|
| AQI | 0.818 |
| PM2.5 | 0.067 |
| PM10 | 0.018 |
| CO | 0.017 |
| AQI_3Day_Avg | 0.012 |
| NO2 | 0.010 |
| O3 | 0.009 |
| AQI_Change | 0.009 |
| SO2 | 0.009 |
| Previous_Day_AQI | 0.007 |

This result is consistent with the short forecasting horizon: the current air-quality state contains substantial information about the following day's state.

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
src/clean_data.py
src/eda.py
src/prepare_ml_data.py
src/train_model.py
src/train_live_model.py
src/evaluate_baseline.py
src/evaluate_predictions.py
src/feature_importance.py
src/city_evaluation.py
src/live_air.py
src/live_forecast.py
```

The project can be run locally with:

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
