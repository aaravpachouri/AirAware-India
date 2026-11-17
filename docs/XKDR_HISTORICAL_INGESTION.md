# XKDR historical ingestion

AirAware can ingest CPCB CAAQM monthly Parquet partitions from the India Air Quality Database maintained by XKDR Forum.

## Credentials

Create an XKDR API key at https://airquality.xkdr.org/signup and configure it as the environment variable `XKDR_API_KEY`. Never commit the key or put it in command-line arguments. The public demo key is limited and cannot read the complete archive.

## Install and run

```bash
pip install -r requirements.txt
# Small, reproducible pilot first:
python scripts/ingest_xkdr_historical.py --start 2024-01 --end 2024-01
# Expand only after inspecting the pilot manifest:
python scripts/ingest_xkdr_historical.py --start 2009-01 --end 2025-09
```

The script downloads one month partition at a time, filters strictly to `source=cpcb_caaqm`, stores each month as a separate Parquet output, and writes a manifest and station registry. The default cache is `.cache/xkdr`; the normalized output is isolated under `data/xkdr_cpcb/`.

## Data integrity rules

- XKDR documents `collected_at` as a naive Asia/Kolkata timestamp. The pipeline retains that wall-time field and adds an explicitly timezone-converted UTC timestamp.
- Original values and unit labels are retained. No conversion is guessed; unrecognized unit labels are flagged.
- Negative/non-finite values and unsupported pollutant names are retained but marked ineligible. The script does not impute, fill gaps, calibrate, or remove outliers.
- Exact duplicates (same station, pollutant, timestamp, unit and value) are collapsed. Conflicting readings are not averaged or silently discarded.
- XKDR station IDs are not mapped to OpenAQ IDs. No joint training occurs.
- Only records flagged `eligible_raw_observation` are candidates for a later, separately validated training dataset. This flag is not regulatory validation.

## Attribution and limitations

Credit: **XKDR Forum, India Air Quality Database**, and the **Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring network**. The database is released under CC BY 4.0. XKDR states that readings are published as received and are not independently validated by XKDR. See https://airquality.xkdr.org/#docs.

This stage does not change the live Streamlit application or production forecast artifacts. Historical ingestion, station reconciliation, AQI reconstruction, and model evaluation are separate follow-on gates.
