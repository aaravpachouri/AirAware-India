# AirAware India 🇮🇳

### Short-Term Air Quality Forecasting & Public Air-Quality Intelligence

AirAware India is a lightweight machine-learning system designed to forecast **next-day air quality** and translate air-quality data into simple, understandable information for communities.

The project combines historical Indian air-quality data, machine-learning forecasting, pollutant analysis, and a live monitoring layer to explore whether relatively lightweight ML models can provide useful short-term air-quality forecasts.

> **Status:** Research prototype / educational project

---

## 🎯 Research Question

> **Can lightweight machine-learning models provide useful short-term air-quality forecasts at local monitoring locations using pollutant and temporal features?**

The project focuses on building a practical forecasting pipeline rather than attempting to replace official air-quality monitoring systems.

---

## 🌍 Why AirAware?

Air-quality datasets often contain large amounts of technical information that is difficult for non-technical users to interpret.

AirAware attempts to bridge that gap by combining:

- 📊 Historical air-quality analysis
- 🤖 Machine-learning forecasting
- 📍 Location-specific monitoring
- 🌫️ Pollutant-level analysis
- 📈 Model evaluation
- 💡 Simple public-facing insights

The goal is to turn:

**Raw environmental data → ML prediction → understandable information**

---

# 🧠 System Overview

```text
Historical Air-Quality Data
          │
          ▼
     Data Cleaning
          │
          ▼
   Feature Engineering
          │
          ▼
  Machine Learning Model
          │
          ▼
   Next-Day AQI Forecast
          │
          ├───────────────┐
          ▼               ▼
 Historical Research   Live Monitoring
                       (OpenAQ)
                          │
                          ▼
                 Estimated Current AQI
                          │
                          ▼
                 Public Insight Layer

🔬 Machine Learning
The main forecasting model is a Random Forest Regressor.
The model predicts:
Next_Day_AQI

using current and historical air-quality information.
Main features
- PM2.5
- PM10
- NO2
- CO
- SO2
- O3
- Current AQI
- Previous-day AQI
- 3-day average AQI
- AQI change
- Day
- Month
- Day of week
- City
Categorical city information is encoded using OneHotEncoder.
📈 Model Performance
The current model was evaluated using a chronological 80/20 train-test split.
This prevents future observations from being used to train the model before evaluating earlier observations.
Current results
Metric	Result
Model	Random Forest Regressor
MAE	16.20 AQI points
R²	0.883


The model therefore achieves a mean absolute error of approximately 16 AQI points on the held-out test set.
R² is a statistical goodness-of-fit metric and should not be interpreted as "88.3% accuracy."

🧪 Baseline Comparison
AirAware also uses a simple persistence baseline:
Tomorrow's AQI = Today's AQI

The baseline achieved:
MAE: 18.39

The machine-learning model achieved:
MAE: 16.20

This provides a more meaningful comparison than evaluating the ML model alone.
The result suggests that the model captures information beyond simply assuming that tomorrow's AQI will be identical to today's.
🔍 Feature Analysis
Feature importance analysis showed that current AQI is the strongest predictor in the current model.
Approximate importance:
Feature	Importance
AQI	0.818
PM2.5	0.067
PM10	0.018
CO	0.017
AQI 3-Day Average	0.012
NO2	0.010
O3	0.009
AQI Change	0.009
SO2	0.009
Previous-Day AQI	0.007


This is expected for short-term forecasting because recent AQI conditions contain strong information about near-future air quality.
🏙️ City-Level Analysis
The historical dataset contains air-quality observations from multiple Indian cities.
AirAware performs city-level analysis to investigate how forecasting performance varies geographically.
Examples of cities represented in the dataset include:
- Delhi
- Ahmedabad
- Mumbai
- Bengaluru
- Hyderabad
- Chennai
- Kolkata
- Lucknow
- Patna
- Gurugram
- Jaipur
- Bhopal
- Chandigarh
- Amritsar
- Guwahati
- Coimbatore
- Ernakulam
- Thiruvananthapuram
- Aizawl
- Shillong
The project also generates city-level evaluation results to identify where the model performs better or worse.
📊 Exploratory Data Analysis
The project includes several exploratory analyses:
AQI Distribution
Shows the distribution of AQI values used for modelling.
Average AQI by City
Compares average air-quality levels across cities.
PM2.5 vs AQI
Explores the relationship between particulate pollution and AQI.
Actual vs Predicted
Compares model predictions against observed next-day AQI values.
Generated visualizations are available in:
reports/

🌐 Live Air-Quality Layer
AirAware also includes a live monitoring component using the OpenAQ API.
The live layer:
1. Discovers available monitoring stations in India
2. Finds recently updated stations
3. Retrieves the latest pollutant measurements
4. Normalizes pollutant units
5. Estimates a CPCB-style AQI from available measurements
6. Generates a short-term forecast
7. Converts the results into a simple public-facing insight
The live system supports selecting individual monitoring stations rather than assuming that one measurement represents an entire city.
⚠️ Important AQI Disclaimer
The live dashboard displays an estimated CPCB-style AQI.
It should not be interpreted as an official CPCB AQI value.
Official AQI calculation involves pollutant-specific averaging periods and monitoring requirements. A single recent measurement does not necessarily satisfy those requirements.
Therefore, AirAware labels its live calculation as:
Estimated CPCB-style AQI

rather than presenting it as an official government measurement.
💡 Public Information Layer
AirAware attempts to make technical air-quality information easier to understand.
Instead of only displaying numbers, the dashboard provides:
- Current estimated AQI
- Main pollutant measurements
- Tomorrow's predicted AQI
- Whether conditions are expected to improve or worsen
- Simple non-medical guidance
The purpose is public information, not medical diagnosis or treatment.
🗂️ Project Structure
AirAware-India/
│
├── app.py
│
├── data/
│   ├── cleaned_city_day.csv
│   └── ml_data.csv
│
├── models/
│   └── *.joblib
│
├── reports/
│   ├── actual_vs_predicted.png
│   ├── aqi_distribution.png
│   ├── average_aqi_by_city.png
│   └── pm25_vs_aqi.png
│
├── src/
│   ├── clean_data.py
│   ├── eda.py
│   ├── prepare_ml_data.py
│   ├── train_model.py
│   ├── train_live_model.py
│   ├── live_air.py
│   ├── live_forecast.py
│   ├── evaluate_baseline.py
│   ├── evaluate_predictions.py
│   ├── feature_importance.py
│   └── city_evaluation.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

🧹 Data Pipeline
The historical dataset originally contains pollutant measurements and AQI information collected across Indian cities.
The preprocessing pipeline:
Raw Dataset
    ↓
Remove invalid AQI values
    ↓
Handle missing observations
    ↓
Sort chronologically by city
    ↓
Create temporal features
    ↓
Create lag features
    ↓
Create rolling AQI features
    ↓
Create next-day target
    ↓
Train ML model

The final cleaned historical dataset contains approximately:
24,307 observations

The modelling dataset contains approximately:
22,981 observations

after constructing valid next-day forecasting examples.
🚀 Running the Project Locally
1. Clone the repository
git clone https://github.com/aaravpachouri/AirAware-India.git
cd AirAware-India

2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate

macOS / Linux
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Configure OpenAQ
Create a .env file:
OPENAQ_API_KEY=your_openaq_api_key_here

Do not commit .env to GitHub.
A template is provided as:
.env.example

▶️ Run the Dashboard
streamlit run app.py

The Streamlit dashboard provides:
Latest Air
Live monitoring station information and estimated current air quality.
Historical Research
Historical city-level forecasting and analysis.
Model Performance
Model metrics and evaluation information.
🧪 Reproducing the ML Pipeline
The major pipeline stages can be run individually.
Clean the dataset
python src/clean_data.py

Perform exploratory analysis
python src/eda.py

Prepare ML data
python src/prepare_ml_data.py

Train the forecasting model
python src/train_model.py

Evaluate against the baseline
python src/evaluate_baseline.py

Evaluate predictions
python src/evaluate_predictions.py

Analyse feature importance
python src/feature_importance.py

Evaluate city-level performance
python src/city_evaluation.py

📦 Model Files
The trained .joblib models are intentionally not stored in this GitHub repository because the files exceed GitHub's normal per-file size limit.
The repository therefore contains the training and evaluation code required to reproduce the models.
Current local models include:
airaware_random_forest_v3.joblib
airaware_live_rf.joblib

These are generated locally by the training pipeline.
⚠️ Limitations
AirAware is a research prototype and has several important limitations.
1. Historical dataset limitations
The historical model depends on the quality and completeness of the underlying dataset.
2. Missing data
Environmental monitoring datasets can contain substantial missing observations.
3. Forecast horizon
The current model focuses on next-day AQI forecasting.
It is not designed for long-term forecasting.
4. Spatial limitations
The historical model works primarily with city-level observations, while the live layer operates at individual monitoring stations.
These are different spatial scales.
5. Live AQI estimation
The live dashboard uses recent measurements and estimates a CPCB-style AQI. It is not an official CPCB measurement.
6. Model generalization
Good performance on the historical test set does not guarantee equally good performance in every city or future environmental condition.
7. Extreme pollution events
Prediction errors can increase during sudden pollution spikes or unusual environmental events.
🔮 Future Work
Potential future research directions include:
- Incorporating meteorological variables
- Testing additional forecasting models
- Comparing Random Forest with gradient boosting and time-series approaches
- Improving station-level forecasting
- Building uncertainty intervals around predictions
- Evaluating seasonal performance
- Creating a larger standardized benchmark dataset
- Adding more Indian monitoring stations
- Improving Hindi/Hinglish public explanations
- Performing longer-term out-of-time validation
- Developing a dedicated mobile/public information interface
📚 Research Direction
AirAware is designed as more than a dashboard.
The project provides a foundation for investigating:
How effective can lightweight machine-learning systems be for localized environmental forecasting when computational resources and data availability are limited?

Future iterations can turn the project into a more rigorous research study through stronger baselines, temporal validation, uncertainty estimation, and broader station-level evaluation.
🛠️ Technology Stack
Programming
- Python
Data & ML
- Pandas
- NumPy
- Scikit-learn
Visualization
- Matplotlib
- Seaborn
Dashboard
- Streamlit
Live Data
- OpenAQ API
Model Storage
- Joblib
Development
- Jupyter
- VS Code
- Git
- GitHub
📄 Data Sources
The historical analysis uses an Indian city-level air-quality dataset containing pollutant measurements and AQI observations.
The live monitoring layer uses:
OpenAQ
https://openaq.org/
AirAware does not claim ownership of third-party datasets or monitoring data.
Please review the respective dataset/API licenses and terms before redistributing or deploying the project commercially.
👨‍💻 Author
Aarav Pachouri
Student • Python Developer • AI/ML Enthusiast
GitHub:
https://github.com/aaravpachouri
⭐ Project Status
Active research / development project
AirAware India is being developed as an open machine-learning project exploring the intersection of:
Machine Learning
      +
Environmental Data
      +
Public Information
      +
Responsible AI

If you find the project useful, consider giving it a ⭐ on GitHub.