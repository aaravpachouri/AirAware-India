# AirAware India: Research Protocol

## Purpose

This protocol defines how AirAware's next-day AQI forecasting claims should be tested. It is intended to make the evaluation reproducible and to prevent choosing a model or split because it produces a more attractive score.

## Primary research question

Can city-level historical air-quality observations support next-calendar-day AQI forecasts that outperform simple baselines on unseen dates, and how much does performance vary across cities?

## Secondary questions

1. Does the Random Forest outperform persistence (tomorrow equals today) and a three-consecutive-day mean?
2. Does performance generalize to cities excluded from model fitting?
3. Which cities and AQI ranges have the largest errors?
4. How sensitive are results to missing pollutant observations and seasonal periods?
5. How different is the historical city-day prediction task from the live station-level forecasting task?

## Data contract

- Target: observed AQI on the next calendar day for the same city.
- Feature cutoff: only observations available on or before the forecast date may be used.
- City/date key: each city must have at most one row per calendar date before lag and rolling features are constructed.
- Lag feature: previous-day AQI is valid only when the previous observation is exactly one calendar day earlier.
- Three-day mean: valid only when all three observations are on consecutive calendar days.
- Missing pollutant values remain missing until a model pipeline imputes them using training-fold statistics.
- Any source checksum or source URL change must be reviewed explicitly rather than silently accepted.

## Evaluation protocol

### A. Chronological holdout

1. Sort and split by unique calendar dates, never by shuffled rows.
2. Keep the target windows disjoint across the train/test boundary.
3. Fit preprocessing only on training data.
4. Report MAE in AQI points and R², alongside test rows and date boundaries.
5. Compare all models on exactly the same test rows.

### B. Baselines

Report at least:
- Persistence: predict tomorrow's AQI as today's AQI.
- Three-consecutive-day mean: average the three observed AQI values ending on the forecast date.
- Candidate machine-learning model(s), fit only on the training period.

A more complex model is not considered better merely because its R² is high. Its MAE must be compared against the baselines on the same rows.

### C. Expanding-window rolling-origin evaluation

Use at least five sequential, contiguous test blocks after an initial training period. For each fold, expand the training window and test only on a later date block. Because the target for feature date D is AQI on D+1, exclude the feature date immediately before each test block from training so training labels cannot overlap the test period. Fit imputers and estimators independently inside each fold. Report per-fold train/test date boundaries, row counts, MAE/RMSE, pooled out-of-fold metrics, persistence and three-day-mean baselines, and per-city errors. This evaluates temporal stability across successive future periods for cities represented in the dataset; it is not a substitute for unseen-city or live-station validation.

### D. City-level generalization

Use a city-held-out split in which no city in the test set appears in training. Report each held-out city's row count and MAE, plus a macro-average across cities. Do not describe this as equivalent to the chronological holdout: it answers a different question.

### E. Error analysis

Report:
- MAE and R² overall;
- MAE by city;
- error by observed AQI band;
- sample counts for each subgroup;
- comparison with each baseline;
- any subgroup with too little data to support a stable conclusion.

Where practical, add bootstrap confidence intervals that resample at the city or date-block level rather than treating every correlated row as independent.

## Live forecast validity

The live dashboard uses recent station measurements, while the historical training dataset contains city-day observations. These are not interchangeable sampling units. The live forecast must therefore be described as experimental until it has been evaluated on time-aligned station observations using a clearly documented target, valid units, freshness checks, and a baseline.

The default dashboard still displays a latest-value snapshot estimate and an experimental forecast to preserve the established workflow; neither is an official CPCB reading. An optional Latest Air research panel queries OpenAQ's hourly-average sensor endpoint over a rolling 24-hour window. PM2.5, PM10, NO2, and SO2 use 24-hour means; CO and O3 require the latest complete consecutive 8-hour mean. Each pollutant needs at least 16 distinct valid hourly observations in the 24-hour window, and an overall time-averaged estimate requires at least three eligible pollutants including PM2.5 or PM10. The panel leaves the existing snapshot and forecast untouched if coverage is insufficient or an API call fails. This is a CPCB-style research gate, not a certification of CPCB compliance or official data. References: [CPCB AQI calculation guidance](https://airquality.cpcb.gov.in/ccr_docs/How_AQI_Calculated.pdf) and [OpenAQ measurement aggregation documentation](https://docs.openaq.org/resources/measurements).

Missing pollutant values must never be silently replaced with zero: zero is a real concentration, not a missing-data marker. If a required live model input is unavailable, the forecast should be withheld with an explanation.

## Reporting rules

- Preserve unsuccessful experiments and explain why they were rejected.
- Do not use R² as a percentage accuracy score.
- Do not report only the best-performing city or date window.
- Distinguish observed measurements, calculated estimates, and model predictions in the interface.
- State the data period, split dates, row counts, baseline scores, and limitations with every headline result.
- Do not claim health outcomes, policy impact, or user impact without independent evidence.

## Current limitations and next experiments

The project has a chronological evaluation, a three-day-mean baseline, calendar-aware rolling features, unit-normalization tests, a dataset card, a deterministic city-held-out evaluation, and five-fold city-group validation across all 26 cities. The fixed six-city holdout shows that neither Random Forest variant beats persistence (22.19 MAE); the city-agnostic Random Forest scores 24.64 and city-aware Random Forest 28.25. A model-family benchmark also evaluates Ridge variants: on the chronological holdout, city-agnostic Random Forest scores 16.20 MAE, city-aware Random Forest 16.41, city-agnostic Ridge 17.86, and city-aware Ridge 18.54, compared with persistence at 18.81. On the same city-held-out split, city-agnostic Ridge is best among the tested ML candidates at 23.49 MAE but still trails persistence at 22.19. The repeated five-fold experiment reports macro-city MAE 20.133 for city-agnostic Random Forest versus 21.818 for persistence, but its paired 95% city-bootstrap interval for improvement is [-0.766, 3.646], which includes zero. No tested model therefore demonstrates a reliable advantage over persistence across unseen cities. The compact live model's city-day chronological evaluation is only a historical proxy (MAE 16.96; persistence MAE 18.81); it is not station-level live validation. Next priorities are prospective same-station forecast validation, stronger out-of-city models that beat persistence with uncertainty intervals, and independent comparison against appropriately time-aligned reference observations.

## Dataset provenance

The raw data is the city_day.csv file from the Air Quality Data in India (2015–2020) dataset. The project downloads a pinned mirror and verifies its SHA-256 checksum before preprocessing. Dataset description and a CC0 1.0 license are listed by the Transformative Urban Mobility Initiative data portal:

- https://portal.transport-data.org/@tumi/urban-air-pollution-india-2015-2020
- Original Kaggle dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- Pinned mirror: https://github.com/govindlabhala-cell/India-Air-Quality-EDA

The mirror is used for reproducibility; it is not claimed to be the original data publisher.


## Rolling-origin results

The expanding-window rolling-origin evaluation is run by `python src/rolling_origin_validation.py`. It uses five sequential test blocks after the initial training period, a one-day target-window gap at each fold boundary, and training-fold-only preprocessing. It compares city-aware and city-agnostic Random Forest models plus city-agnostic Ridge against persistence and the three-day mean. The workflow publishes pooled, per-fold, and per-city metrics as CI artifacts. No result is asserted in this protocol until the workflow has completed on the merged commit; the report must preserve every fold and candidate rather than selecting a favourable window.
