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

> Can lightweight machine-learning models provide useful next-day AQI forecasts from historical city-level Indian observations, and how well do they generalize to later dates and unseen cities?

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

A deterministic group split held out six complete cities (4,508 observations) from model fitting; the remaining 20 cities formed the training set. This is a separate experiment from the chronological holdout.

| Held-out city | Rows | City-aware RF MAE | City-agnostic RF MAE | Persistence MAE |
|---|---:|---:|---:|---:|
| Ahmedabad | 555 | 103.27 | 73.59 | 41.42 |
| Chennai | 1,850 | 22.39 | 22.15 | 24.00 |
| Coimbatore | 286 | 8.02 | 8.18 | 6.74 |
| Ernakulam | 140 | 7.33 | 7.37 | 6.91 |
| Jorapokhar | 660 | 22.81 | 23.55 | 28.52 |
| Thiruvananthapuram | 1,017 | 10.08 | 10.19 | 10.75 |
| **All held-out rows (weighted MAE)** | **4,508** | **28.25** | **24.64** | **22.19** |

The city-aware model had R² 0.545; the city-agnostic variant had R² 0.703. The three-day mean baseline MAE was 24.64. Removing the city feature improved the aggregate Random Forest score, but neither model beat persistence on this held-out-city split. Ahmedabad was especially difficult for both models. This fixed split does not prove performance for every unseen city, but it demonstrates that the temporal holdout score must not be generalized to cities outside the training set.

### Repeated city-group validation with paired uncertainty intervals

A five-fold GroupKFold experiment evaluated every candidate on held-out cities. Each of the 26 cities appears in exactly one test fold; all predictions for a city are combined before its MAE is calculated. The reported macro-city MAE gives each city equal weight, rather than allowing the largest cities to dominate. For each model, the paired improvement is defined as persistence MAE minus model MAE, so positive values favour the model. A 95% percentile bootstrap interval was calculated by resampling the 26 city-level paired differences (10,000 replicates; fixed random seed).

| Model | Macro-city MAE | Median city MAE | Mean improvement vs persistence | 95% CI for improvement |
|---|---:|---:|---:|---:|
| Random Forest (city-aware) | 20.594 | 18.135 | +1.224 | [-2.624, 3.878] |
| Random Forest (city-agnostic) | 20.133 | 18.338 | +1.685 | [-0.766, 3.646] |
| Ridge (city-aware) | 22.718 | 20.351 | -0.901 | [-2.744, 0.527] |
| Ridge (city-agnostic) | 21.736 | 19.901 | +0.082 | [-1.267, 1.182] |
| Persistence | 21.818 | 21.568 | — | — |
| Three-day mean | 25.523 | 25.896 | — | — |

**Interpretation:** the city-agnostic Random Forest has the lowest macro-city MAE among the tested ML candidates, but every model's 95% interval for paired improvement includes zero. This experiment therefore does not establish a reliable improvement over persistence across unseen cities. The small positive point estimates are exploratory, not evidence of a statistically clear advantage. This result is stricter than selecting a model from one fixed six-city holdout and should remain visible alongside the chronological result. The 26-city folds are a group-held-out evaluation on this historical dataset, not prospective validation of current stations.

### Live AQI aggregation guard

The Latest Air page includes an optional coverage check based on OpenAQ's hourly sensor averages. It reports time-averaged pollutant values only after checking distinct hourly coverage and the pollutant-specific averaging window. PM2.5, PM10, NO2, and SO2 use 24-hour means; CO and O3 use a complete 8-hour rolling mean. The minimum data gate requires 16 distinct valid hourly observations in the recent 24-hour window, and an overall estimate requires at least three eligible pollutants including PM2.5 or PM10. This optional check does not replace the default live snapshot or forecast, and its output is an OpenAQ-derived research estimate—not an official CPCB value. The live station forecast still needs prospective station-aligned evaluation.

### Separate live forecast model — historical proxy evaluation

The live dashboard's model is trained on six features: current AQI, PM2.5, PM10, day, month, and day of week. The training script now uses a chronological split and a median imputer fitted only on the training rows. Metrics below describe a held-out historical city-day task, not future observations from a selected live station.

| Evaluation detail | Result |
|---|---:|
| Training rows | 14,232 |
| Held-out test rows | 8,220 |
| Last training feature date | 2019-05-24 |
| First test feature date | 2019-05-26 |
| Test MAE | 16.96 AQI points |
| Test R² | 0.877 |
| Persistence baseline MAE | 18.81 AQI points |
| Three-day mean baseline MAE | 22.70 AQI points |
| Deployment model size | 36.4 MiB |

On this historical test, the compact live-model pipeline reduced MAE by approximately 9.8% relative to persistence. The deployable model is refit on all eligible historical rows only after the held-out metrics are calculated. It uses 100 trees with maximum depth 16 and minimum leaf size 3 to reduce memory footprint compared with the previous 318 MiB artifact.

This evaluation does not remove the domain gap between historical city-day AQI and current station-level estimates. The live forecast must remain labelled experimental until it is validated against time-aligned subsequent measurements from the same stations.

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

The live estimate does not necessarily reproduce the official averaging procedure used for government AQI reporting. OpenAQ unit conversion uses conventional 25 °C / 1 atm approximations for gas measurements reported in ppb or ppm.

### Forecast limitations

The persistence baseline is strong, showing that short-horizon air-quality forecasting is inherently challenging. The separate live station forecast uses a different model and input distribution; it has not yet been validated against subsequent measurements from the selected station, so the historical city-day metrics do not establish live forecast accuracy.

### Distribution limitations

Model performance varies between cities and pollution regimes.

### Operational limitations

Live station measurements can be delayed or unavailable.

---

## 15. Model-family benchmark and current evidence

A reproducible benchmark compared city-aware and city-agnostic Random Forest and Ridge regression pipelines. All candidates were evaluated on identical rows within each fixed split, with imputation (and Ridge scaling) fitted inside each training pipeline.

### Chronological holdout

| Candidate | MAE (AQI points) | R² |
|---|---:|---:|
| City-agnostic Random Forest | **16.20** | 0.895 |
| City-aware Random Forest | 16.41 | 0.890 |
| City-agnostic Ridge | 17.86 | 0.869 |
| City-aware Ridge | 18.54 | 0.868 |
| Persistence | 18.81 | — |
| Three-day mean | 22.70 | — |

### Fixed unseen-city holdout

| Candidate | MAE (AQI points) | R² |
|---|---:|---:|
| City-agnostic Ridge | **23.49** | 0.782 |
| City-aware Ridge | 24.72 | 0.757 |
| City-agnostic Random Forest | 24.64 | 0.703 |
| City-aware Random Forest | 28.25 | 0.545 |
| Persistence | **22.19** | — |
| Three-day mean | 24.64 | — |

The fixed city holdout includes Ahmedabad, Chennai, Coimbatore, Ernakulam, Jorapokhar, and Thiruvananthapuram (4,508 test observations). The city-agnostic Ridge model is the strongest machine-learning candidate on this particular holdout, but it does not beat persistence. Ahmedabad is a major error contributor, especially for the Random Forest candidates. This indicates a transfer problem rather than a solved cross-city forecasting task.

The 0.21-point MAE difference between the two Random Forest candidates on the chronological split is not enough to justify a model replacement without repeated temporal folds and paired uncertainty intervals. The deployed model is therefore not changed by this benchmark.

### Live AQI validity gate

The current live implementation applies pollutant breakpoints to the latest available sensor values. CPCB guidance specifies 24-hour averages for most pollutants, 8-hour averages for CO and O3, at least 16 hours of data for a pollutant sub-index, and at least three pollutants (including PM2.5 or PM10) for an overall AQI. A latest-value snapshot cannot establish these conditions. AirAware must continue to label its live calculation as an estimate and should withhold an overall AQI when the required averaging windows and coverage cannot be verified. OpenAQ exposes hourly and daily aggregates and coverage metadata that can support a future compliant implementation, but the application must validate the returned time windows and completeness rather than assume they are sufficient.

## 16. Future Research

Possible extensions include:

- repeated city-held-out splits and model-family benchmarks
- station-aligned validation of live forecasts
- station-specific models
- meteorological features
- probabilistic forecasts
- prediction intervals
- multi-day forecasting
- improved missing-data handling
- calibration of forecast uncertainty
- broader public-impact evaluation

---

## 17. Reproducibility

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
src/evaluate_city_holdout.py
src/evaluate_predictions.py
src/feature_importance.py
src/city_evaluation.py
src/live_air.py
src/live_forecast.py
src/live_forecast_features.py
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

## 18. Conclusion

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

## 19. Prospective live-station evaluation

Historical city-day scores and live station forecasts are different prediction tasks. To measure the deployed workflow directly, AirAware maintains an append-only prospective ledger. Each eligible forecast is issued for a specific OpenAQ location, records its issue timestamp and model artifact hash, and is evaluated only against a later observation from the same location ID.

### Predefined pairing rule

- Forecast horizon: nominally 24 hours.
- Eligible observation window: 18–30 hours after issue time.
- Station identity: exact OpenAQ location ID; no cross-station substitution.
- Eligibility at issue: latest station measurement no older than six hours and at least three available pollutant sub-indices, including PM2.5 or PM10.
- Baseline: persistence, evaluated against the exact same later observation.
- Reported metrics: paired MAE/RMSE, signed bias, median and P90 absolute error, model wins/losses/ties, actual horizon, and per-station/per-city sample counts.
- Uncertainty: exploratory station-cluster bootstrap interval only when at least five distinct stations have evaluated pairs.

The collection workflow runs twice daily at 01:17 and 13:17 UTC. This cadence is designed around the fixed 18–30 hour matching window; a once-daily schedule can miss forecasts issued manually or at other times. Each run publishes a JSON summary artifact and appends new forecast/evaluation records to the public ledger.

### Initial collection status (10 November 2026)

The first successful collection recorded **25 pending forecasts across 25 distinct station IDs**. At that snapshot, **zero forecasts had matured into evaluated same-station pairs**, so no prospective MAE, RMSE, or model-versus-persistence conclusion was available. These pending forecasts are a collection milestone, not an accuracy result.

The live validation study must remain open until enough later observations have been paired. Report every result with its sample size, station and city coverage, date range, actual forecast horizon, and paired persistence comparison. If the model does not outperform persistence, that is a valid research result and should be reported rather than hidden.

Protocol and live ledger:
- [Prospective station validation protocol](PROSPECTIVE_VALIDATION.md)
- [Public prospective validation ledger](../data/prospective_station_validation.jsonl)

The prospective target is an OpenAQ-derived station snapshot based on available pollutant sub-indices. It is **not official CPCB AQI**, not a regulatory 24-hour average, and not an independently certified reference measurement. The study tests the live pipeline's consistency against subsequent same-provider observations; it does not establish health, regulatory, or public-policy impact.

## 20. Expanding-window rolling-origin evaluation

A new temporal robustness experiment evaluates the next-day model across five sequential test blocks using an expanding training window. Unlike a single 80/20 split, this measures whether performance is stable across multiple later periods. At each fold, the feature date immediately preceding the test block is excluded from training because its next-day label would overlap the first test date. Preprocessing is refit within each training fold. The experiment reports pooled out-of-fold metrics, each fold's date boundaries and row counts, and per-city results, comparing city-aware and city-agnostic Random Forest, city-agnostic Ridge, persistence, and the three-day mean.

This is a temporal robustness test for cities represented in the historical dataset. It does not establish generalisation to unseen cities, station-level performance, or official CPCB AQI accuracy. The generated files are `reports/rolling_origin_summary.json`, `reports/rolling_origin_fold_metrics.csv`, and `reports/rolling_origin_city_metrics.csv`; the CI workflow retains them as artifacts.


### Measured rolling-origin results (10 November 2026)

The merged CI evaluation completed five expanding-window folds covering **16,577 out-of-fold rows**. Each candidate and baseline was evaluated on the same pooled test rows. Preprocessing was fitted independently within each training fold, and a one-day target gap was maintained at fold boundaries.

| Model | Pooled MAE | Pooled RMSE | MAE improvement vs persistence |
|---|---:|---:|---:|
| Persistence | 22.1571 | 35.8745 | — |
| Three-day mean | 26.3910 | 40.9693 | -4.2339 |
| Random Forest (city-aware) | **19.3747** | **29.9239** | **+2.7824** |
| Random Forest (city-agnostic) | 19.5583 | 30.1022 | +2.5988 |
| Ridge (city-agnostic) | 21.1455 | 32.0839 | +1.0116 |

On this temporal evaluation, the city-aware Random Forest reduced pooled MAE by approximately **12.6%** relative to persistence (2.7824 / 22.1571). Its pooled MAE was 0.1836 points lower than the city-agnostic Random Forest. The three-day mean baseline performed worse than persistence.

Fold-level results show that performance was not uniform across time. The city-aware Random Forest beat persistence in folds 1, 3, 4, and 5, but not fold 2 (MAE 20.6466 versus 20.4074). The city-agnostic Random Forest also failed to beat persistence in fold 2. These differences should remain visible; pooled improvement does not mean every period or city benefits.

**Interpretation and limits:** this is evidence of improved temporal performance across later dates for cities represented in the historical dataset. It is not unseen-city generalisation, station-level prospective validation, or evidence of official CPCB AQI accuracy. The previously reported held-out-city experiments remain important: the models did not beat persistence on the fixed unseen-city holdout, and repeated city-group uncertainty intervals included zero. Therefore, do not claim that AirAware generalises reliably to new cities or that the live station forecast is validated. Prospective same-station pairs are still required for that claim.

The complete fold and per-city metrics are retained in the GitHub Actions artifact `airaware-regenerated-model` for the successful evaluation run. The summary and CSV files are also generated by `src/rolling_origin_validation.py`.
