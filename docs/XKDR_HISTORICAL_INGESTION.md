# XKDR historical ingestion and pilot QA

AirAware ingests CPCB CAAQM monthly Parquet partitions from the India Air Quality Database maintained by XKDR Forum.

## Credentials

Create an XKDR API key at https://airquality.xkdr.org/signup and configure it as the GitHub Actions secret `XKDR_API_KEY`. Never commit the key or put it in command-line arguments.

## Run a pilot

Use Actions → **XKDR historical ingestion tests and pilot** → Run workflow. Start with one month, such as `2024-01`. The workflow ingests the selected partition, runs an independent QA audit, and uploads both the data and report.

## Independent QA report

Run locally after ingestion:

```bash
python scripts/audit_xkdr_pilot.py --data-dir data/xkdr_cpcb
```

The audit checks the Parquet schema, row count against the ingestion manifest, CPCB source isolation, missing identifiers/timestamps/units, IST-to-UTC consistency, duplicate keys remaining after ingestion, parameter/unit distributions, station registry reconciliation, coordinate ranges, and broad value-range diagnostics.

The report is diagnostic and does not rewrite or discard observations. Broad plausibility ranges are warnings only; they are not unit-aware regulatory thresholds. The audit cannot reconstruct duplicates already collapsed by the ingestion step, so its duplicate check applies to the deduplicated output only.

## Training gate

Do not use the ingestion `eligible_raw_observation` flag as a training approval. Before model use, separately:
- normalize units by pollutant and retain original values;
- validate parameter aliases and source measurement semantics;
- reconcile station identities using verified metadata/coordinates and temporal validity;
- define pollutant-specific quality rules and missingness handling;
- reconstruct CPCB AQI only when the required pollutant coverage and averaging periods are available;
- compare the resulting dataset against the existing training set and evaluate chronological and station-held-out baselines.

No historical XKDR record is currently merged into OpenAQ live history or the production model by this workflow.

## Attribution and limitations

Credit: **XKDR Forum, India Air Quality Database**, and the **Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring network**. The database is released under CC BY 4.0. XKDR states that readings are published as received and are not independently validated by XKDR. See https://airquality.xkdr.org/#docs.
