# AirAware India

> **Machine learning for short-term air-quality forecasting and understandable environmental insights.**

AirAware India is a research-oriented environmental ML platform that combines historical Indian air-quality data, short-term AQI forecasting, live monitoring-station data, and a public-facing dashboard.

## Why AirAware?

Air-quality datasets are useful for research, but raw pollutant measurements are difficult for most people to interpret.

AirAware explores whether a lightweight machine-learning model can provide useful **next-day AQI forecasts** while also translating air-quality measurements into simple, understandable environmental information.

## Research Question

> **Can lightweight machine-learning models provide useful short-term air-quality forecasts at local monitoring locations using pollutant and temporal features?**

## What It Does

- Live monitoring-station discovery across India
- Latest pollutant measurements through OpenAQ
- Estimated CPCB-style AQI calculation
- Next-day AQI forecasting with Random Forest
- Historical city-level air-quality exploration
- Pollutant-level breakdowns
- Forecast evaluation
- Model-performance reporting
- Public-facing environmental insights
- Interactive Streamlit dashboard with animated UI

## System Architecture

```text
Historical Indian AQI Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest Forecasting Model
        │
        ├──────────────► Model Evaluation
        │
        ▼
                 AirAware Dashboard
                        ▲
                        │
                 OpenAQ Monitoring
                        │
                        ▼
                 Latest Pollutants
                        │
                        ▼
              Estimated CPCB-style AQI
                        │
                        ▼
                 Short-term Forecast
                        │
                        ▼
              Environmental Insight
```

## Live Demo

**Public app:** https://airaware-india-s5nokzxaczphwtyzwhscug.streamlit.app/

The Streamlit Community Cloud app was verified in a private/incognito browser session without a sign-in prompt. Live observations and forecasts depend on the availability and freshness of the upstream monitoring data.

## Dashboard

### Latest Air

![AirAware Latest Air](docs/screenshots/latest-air.png)

### Historical Research

![AirAware Historical Research](docs/screenshots/historical-research.png)

### Model Performance

![AirAware Model Performance](docs/screenshots/model-performance.png)

## Machine Learning

### Target

The main forecasting task predicts:

**Next_Day_AQI**

### Historical model

**Random Forest Regressor**

The final historical model uses:

- City
- PM2.5
- PM10
- NO2
- CO
- SO2
- O3
- Current AQI
- Previous-day AQI
- 3-day AQI average
- AQI change
- Day
- Month
- Day of week

### Reproducible chronological evaluation

GitHub Actions rebuilt the dataset from a checksum-verified raw CSV and retrained the Random Forest using a date-based split and training-only pollutant imputation.

| Metric | Leakage-controlled temporal evaluation |
|---|---:|
| Training rows | 14,597 |
| Held-out test rows | 8,368 |
| Last training feature date | 2019-05-24 |
| First test feature date | 2019-05-26 |
| Test MAE | **16.21 AQI points** |
| Test R² | **0.898** |
| Persistence baseline MAE | **18.94 AQI points** |

The model's MAE is about 14.4% lower than the persistence baseline on the same held-out rows. Performance varies across cities; the result is specific to this historical dataset and is not a claim about current air quality.

**Why the older model score is not used:** the previous checked-in artifact scored MAE 11.39 and R² 0.938 on a later-date holdout, but its original training cutoff is undocumented. It may have seen future dates during training, so that score is not accepted as reliable out-of-sample evidence.

**Preprocessing method:** pollutant missing values are retained in the prepared dataset and imputed inside the model pipeline using medians fitted only on the training rows. The raw source file is downloaded from a public mirror and verified against SHA-256 `0d84b21c3e4878bbad8df362f2ab05f61ad959538dddf5918e714077ed3c1847` before preprocessing. The source CSV itself is not committed to the repository.

The baseline is a persistence model:

```text
Tomorrow's AQI = Today's AQI
```

**Important:** R² is not a percentage accuracy score.

### Reproduce the historical evaluation

Install the dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the full reproducible pipeline:

```bash
python src/download_source_data.py
python src/clean_data.py
python src/prepare_ml_data.py
python src/train_model.py
python src/evaluate_baseline.py
python src/evaluate_predictions.py
python src/city_evaluation.py
python src/feature_importance.py
```

To evaluate an existing compatible model artifact against the prepared dataset, run `python src/evaluate_saved_model.py` before retraining. The downloader verifies the source CSV checksum and fails closed if the file differs. The training pipeline learns pollutant imputation values only from training-period rows; the held-out test period is not used to fit the imputer.

## Feature Importance

The leakage-controlled training run identified current AQI as the dominant predictor (importance 0.8068), followed by PM2.5 (0.0718). Feature importance is descriptive, not causal.

This is consistent with the short-horizon nature of the forecasting problem: today's air-quality state contains substantial information about the following day's conditions.

## Live Monitoring

AirAware uses **OpenAQ** to discover Indian monitoring stations and retrieve recent pollutant observations.

The live pipeline:

```text
OpenAQ
  ↓
Indian monitoring stations
  ↓
Selected station
  ↓
Latest pollutant observations
  ↓
Unit normalization
  ↓
Estimated pollutant sub-indices
  ↓
Estimated CPCB-style AQI
  ↓
AirAware forecast
```

### Important live-data limitation

The live value displayed by AirAware is an **estimated CPCB-style AQI**, not an official CPCB AQI.

Official AQI calculations use prescribed averaging periods and sufficient observations. A latest station measurement may not satisfy those requirements.

The dashboard therefore clearly treats this component as a research prototype.

## Historical Data

The historical model was developed using Indian city-level air-quality observations from **2015–2020**.

The data-processing pipeline includes:

1. Removing invalid dates and unusable AQI records
2. Retaining missing pollutant observations for train-only imputation
3. Normalizing pollutant columns
4. Creating chronological city-level features
5. Creating previous-day AQI
6. Creating a rolling 3-day AQI average
7. Creating AQI change
8. Creating calendar features
9. Creating a true next-day prediction target

## Project Structure

```text
AirAware-India/
│
├── app.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── cleaned_city_day.csv
│   └── ml_data.csv
│
├── docs/
│   └── screenshots/
│
├── models/
│   ├── airaware_random_forest_v3.joblib
│   └── airaware_live_rf.joblib
│
├── reports/
│   ├── actual_vs_predicted.png
│   ├── aqi_distribution.png
│   ├── average_aqi_by_city.png
│   └── pm25_vs_aqi.png
│
└── src/
    ├── download_source_data.py
    ├── clean_data.py
    ├── eda.py
    ├── prepare_ml_data.py
    ├── train_model.py
    ├── train_live_model.py
    ├── evaluate_saved_model.py
    ├── evaluate_baseline.py
    ├── evaluate_predictions.py
    ├── feature_importance.py
    ├── city_evaluation.py
    ├── live_air.py
    └── live_forecast.py
```

## Tech Stack

**Machine Learning**
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

**Data**
- Historical Indian air-quality data
- OpenAQ API

**Application**
- Streamlit
- HTML/CSS
- JavaScript
- Three.js
- GSAP

## Run Locally

### 1. Clone

```bash
git clone https://github.com/aaravpachouri/AirAware-India.git
cd AirAware-India
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure OpenAQ

```powershell
copy .env.example .env
```

Add your own key:

```env
OPENAQ_API_KEY=your_openaq_api_key
```

Never commit `.env`.

### 5. Run the application

```powershell
streamlit run app.py
```

## Research Pages

### Latest Air

Shows:

- Selected monitoring station
- Latest pollutant measurements
- Estimated AQI
- AQI category
- Tomorrow's predicted AQI
- AQI movement
- Pollutant sub-indices
- Environmental guidance

### Historical Research

Allows exploration of:

- Historical city/date observations
- Current AQI
- Predicted next-day AQI
- Actual next-day AQI where available
- Prediction error
- Recent AQI trend

### Model Performance

Documents:

- Model type
- MAE
- R²
- Feature set
- Research limitations

## Reports

The `reports/` directory contains the main research visualizations:

- AQI distribution
- Average AQI by city
- PM2.5 vs AQI
- Actual vs predicted AQI

## Limitations

AirAware is intentionally presented as an experimental research prototype.

Current limitations include:

- Historical training data is city-level
- Live monitoring data can be delayed
- Live point measurements do not necessarily satisfy official AQI averaging requirements
- The historical model is optimized for short-term forecasting rather than long-range prediction
- Performance varies by city
- A strong persistence baseline means short-term AQI prediction is inherently challenging

## Future Research

Potential extensions include:

- More rigorous temporal and city-level validation
- Additional meteorological features
- Stronger baseline comparisons
- Probabilistic forecasting
- Multi-step forecasting
- Better handling of missing observations
- Station-specific modelling
- Broader public-health communication research

## Portfolio Context

AirAware India is part of a broader progression of AI/ML projects:

```text
JARVIS X
Software Engineering + AI Agents
        ↓
RAG Model
Retrieval + NLP + Multimodal AI
        ↓
AirAware India
Machine Learning + Forecasting + Environmental Impact
        ↓
Future Flagship AI Model
Model Training + Evaluation + Open Research
```

## Status

**Current status: Public research prototype**

The project is deployed as an interactive Streamlit application and maintained as an open-source research/engineering project.

## License

Add the project's final license here before publication.

---

**Built by Aarav Pachouri**
