# Prospective same-station forecast validation

## Purpose

Historical evaluation uses city-day records and cannot establish how the live dashboard performs at an individual monitoring station. This collector starts a prospective evidence ledger: issue a 24-hour forecast for an OpenAQ location, then compare it with a later eligible observation from that exact location.

## Collection

GitHub Actions runs daily at 01:17 UTC and can also be started manually. It discovers up to 25 recently reporting Indian stations and records forecasts only when:
- The station's latest reported measurement is no more than six hours old.
- At least three pollutant sub-indices are available, including PM2.5 or PM10.
- The live model artifact exists and the feature builder accepts the inputs.

A later run pairs a pending forecast only with the same OpenAQ location ID and a measurement timestamp 18–30 hours after issue time. It records model absolute error and persistence-baseline absolute error on the same paired target.

The ledger records public station metadata, timestamps, pollutant coverage names, current/predicted AQI, model SHA-256 and evaluation provenance. It never records API keys.

## Setup

1. Open repository Settings → Secrets and variables → Actions.
2. Add a repository secret named OPENAQ_API_KEY containing the OpenAQ v3 API key.
3. Open Actions → Prospective station validation and run it once with Run workflow.
4. Check the run log for discovered stations, eligible snapshots, created forecasts and matured pairings.

The secret must never be committed or pasted into chat. Collection fails closed if it is absent.

## Public ledger

Records are stored as JSON Lines in data/prospective_station_validation.jsonl. Public commit history provides an auditable record of additions and pairings. Do not manually edit historical evaluated rows.

## Interpretation and limitations

- The model was trained on historical city-day data; this is a prospective station-level deployment test, not a claim that the model is station-trained.
- The target is an OpenAQ-derived station snapshot calculated as the maximum available CPCB-style pollutant sub-index. It is not an official CPCB AQI and is not a prescribed regulatory 24-hour average.
- OpenAQ latest returns the latest available value per sensor; pollutant timestamps may differ. The ledger stores the station's latest overall timestamp and available sub-index names, so this is not equivalent to a fully synchronised multi-pollutant sample.
- A station is eligible only when station-level age is at most six hours and at least three sub-indices including particulate matter are present. These safeguards reduce, but do not eliminate, temporal mismatch.
- The first collection creates pending forecasts. Evaluated metrics remain empty until later runs retrieve eligible observations 18–30 hours after issue time.
- A failed or missed daily run can leave a forecast unmatched. Do not widen the matching window after seeing results.
- Results are descriptive. Compare model MAE with persistence MAE on the same evaluated records and report sample size, station/city coverage and time period. Do not claim broad generalisation from a small or geographically biased sample.
- OpenAQ attribution and applicable data terms must be respected. The public ledger contains station metadata and derived research values; no API credential is stored.

## Reproducible summary

Run the following command:

    python src/evaluate_prospective_station_validation.py

The script reports zero evaluated forecasts until actual same-station pairs exist. It never substitutes historical validation metrics for prospective results.


## Statistical reporting (prospective validation)

The summary script reports the model and persistence MAE/RMSE on exactly the same evaluated forecast-observation pairs, signed mean bias, median and 90th-percentile absolute error, paired model wins/losses/ties, actual forecast-horizon range, and per-station/per-city sample counts and errors.

When at least five distinct stations have evaluated records, it also calculates an exploratory paired percentile interval for MAE improvement using 2,000 bootstrap replicates that resample whole station clusters. This preserves repeated observations from a station within each resampled cluster. The interval is not a substitute for a pre-registered sample-size target or an independent geographic holdout; it must be reported alongside the number of stations, cities, and paired forecasts. With fewer than five stations, the interval is omitted.

A positive MAE improvement means persistence MAE minus model MAE is positive and therefore favours the model. The interval containing zero means the available prospective sample does not establish a reliable improvement. Do not convert these descriptive station-snapshot metrics into official CPCB AQI accuracy claims.
