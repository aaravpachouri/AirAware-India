# AirAware India — Portfolio & Research Brief

**Project type:** Independent environmental machine-learning research and deployed software prototype  
**Repository:** https://github.com/aaravpachouri/AirAware-India  
**Live application:** https://airaware-india-s5nokzxaczphwtyzwhscug.streamlit.app/  
**Primary stack:** Python, Streamlit, Pandas, NumPy, scikit-learn, OpenAQ API, GitHub Actions

## One-sentence summary

AirAware India is an open-source research-oriented platform that explores next-day air-quality forecasting from historical Indian city-level observations and connects that research to an interactive dashboard for recent monitoring-station data.

## Problem and research question

Raw air-pollution measurements are difficult for many people to interpret, while short-horizon AQI forecasts are challenging because air quality is persistent, seasonal, geographically variable, and affected by missing or delayed measurements.

The project's research question is:

> Can lightweight machine-learning models provide useful next-day AQI forecasts from historical city-level Indian observations, and how well do they generalize to later dates and unseen cities?

## What was built

- A Streamlit application with live monitoring-station discovery, pollutant readings, estimated CPCB-style AQI, experimental next-day forecasts, historical exploration, and model-performance reporting.
- A reproducible data pipeline for checksum-verified source retrieval, cleaning, chronological feature engineering, next-calendar-day target construction, and training-only imputation.
- A Random Forest forecasting pipeline with persistence and three-day-mean baselines.
- Multiple evaluation designs that answer different questions: chronological holdout, expanding-window rolling origin, fixed unseen-city holdout, repeated city-group validation with paired uncertainty intervals, and prospective same-station validation infrastructure.
- A prospective evidence collector that records forecasts and later attempts to pair them with observations from the exact same OpenAQ station ID.

## Measured machine-learning results

### Expanding-window rolling-origin test

Five sequential test folds produced 16,577 out-of-fold rows.

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence | 22.1571 | 35.8745 |
| Three-day mean | 26.3910 | 40.9693 |
| Random Forest (city-aware) | **19.3747** | **29.9239** |
| Random Forest (city-agnostic) | 19.5583 | 30.1022 |
| Ridge (city-agnostic) | 21.1455 | 32.0839 |

The city-aware Random Forest reduced pooled MAE by approximately 12.6% relative to persistence on these rolling-origin rows. This improvement was not uniform across every fold.

### Important counter-result: unseen cities

On a separate fixed six-city holdout (4,508 observations), persistence achieved MAE 22.19, compared with 24.64 for the city-agnostic Random Forest and 28.25 for the city-aware Random Forest. Repeated city-group validation also produced paired uncertainty intervals that included zero.

**Interpretation:** temporal performance on represented cities is encouraging, but reliable generalisation to unseen cities has not been established. This negative result is part of the research contribution and should be disclosed, not hidden.

### Historical live-model proxy

The compact live forecast model achieved historical city-day MAE 16.96 versus persistence MAE 18.81 on its chronological proxy test. This does not establish live station forecast accuracy.

## Scientific integrity and limitations

- The live value is an OpenAQ-derived estimate, not an official CPCB observation or regulatory AQI certification.
- Latest pollutant readings may have different timestamps; they are not necessarily a synchronised sample.
- The prospective target is a later OpenAQ station snapshot, not a prescribed regulatory 24-hour average.
- Prospective validation requires a later eligible observation from the same station ID. A forecast is pending until that pair exists.
- At the last reviewed ledger inspection on 10 November 2026, 25 forecasts were pending and zero were evaluated. These are a time-stamped historical snapshot, not a current live count.
- No health outcome, policy effect, adoption, or user impact is claimed without independent evidence.
- The Streamlit deployment's edge probe encountered a redirect sequence through Streamlit's auth handoff. This probe did not establish whether a normal browser session can access the app; manual browser verification remains required.
- The repository does not currently declare a software license. Do not imply that reuse rights are granted until a license is selected.

## Reproducibility

The repository includes the research protocol, dataset card, tests, training/evaluation scripts, and GitHub Actions workflows. The successful main-branch AQI evaluation workflow on 10 November 2026 compiled the application and research scripts, passed the complete test suite, rebuilt the checksum-verified dataset, retrained the models, and generated rolling-origin reports.

Useful entry points:

- [Research protocol](RESEARCH_PROTOCOL.md)
- [Dataset card](DATA_CARD.md)
- [Full research report](AirAware_Research_Report.md)
- [Prospective validation protocol](PROSPECTIVE_VALIDATION.md)
- [Rolling-origin evaluator](../src/rolling_origin_validation.py)

Reproduce the historical pipeline using the commands in the repository README. Prospective collection additionally requires an OpenAQ API key stored as a GitHub Actions repository secret; never commit the key.

## Suggested portfolio / application description

> I built AirAware India, an environmental machine-learning research platform that combines a live monitoring dashboard with reproducible next-day AQI forecasting experiments. I designed a leakage-aware data pipeline, compared Random Forest and Ridge models against persistence baselines, and implemented chronological, expanding-window, and unseen-city evaluation. The rolling-origin experiment showed a 12.6% pooled MAE reduction over persistence across 16,577 out-of-fold observations, while the unseen-city tests revealed that this improvement did not reliably transfer to new cities. I also built a prospective same-station validation pipeline so live forecasts can be tested against later observations rather than relying only on historical proxies.

This description is deliberately specific about both the positive result and the generalisation limitation.

## Next evidence to collect

1. Verify the public app manually in a clean browser session, including first load, station selection, stale-data handling, research coverage check, and mobile layout.
2. Capture fresh screenshots of the actual deployed build for the portfolio.
3. Allow the scheduled prospective collector to accumulate eligible same-station pairs; report sample size and uncertainty alongside paired model-versus-persistence metrics.
4. Obtain genuine user feedback or usage evidence only if it can be collected ethically and independently.
5. Select and add a repository license before inviting reuse or contributions.
