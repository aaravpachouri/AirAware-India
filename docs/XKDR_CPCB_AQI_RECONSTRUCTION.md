# XKDR CPCB unit normalization and AQI reconstruction

## Scope and safety

This is a separate research derivation from the isolated XKDR CPCB CAAQM Parquet files. It does not edit raw observations, alter OpenAQ data, feed the live app, or train/update any forecast model. Outputs are written to `data/xkdr_cpcb_derived/`.

The derived values must be described as **reconstructed CPCB-methodology AQI from XKDR-republished observations**, not as official CPCB-published AQI. XKDR states that readings are published as received and are not independently validated by XKDR.

## Unit policy

The pipeline canonicalizes spelling only:
- `µg/m³`, `μg/m³`, `ug/m3`, `µg/m3`, `μg/m3` → `ug/m3`
- `mg/m³`, `mg/m3` → `mg/m3`

CPCB AQI breakpoints use µg/m³ for PM2.5, PM10, NO2, SO2, O3, NH3 and Pb, and mg/m³ for CO. The pipeline does not perform ppb/ppm-to-mass conversion because the XKDR rows do not carry measurement temperature and pressure needed to make that conversion explicit and reproducible. The January 2024 pilot's CO observations are already mg/m³; its other supported AQI pollutants are µg/m³. NOx is ppb in this pilot but is not one of the eight National AQI pollutants and is not substituted for NO2.

Raw source values and units remain unchanged. NO, NOx, benzene, toluene, xylene and ethylbenzene remain available in the raw Parquet source but are not included in CPCB NAQI reconstruction. Lead is one of the eight CPCB pollutants but is absent from this pilot's parameter set; no lead values are invented or imputed.

## CPCB AQI calculation

CPCB describes pollutant sub-indices as linear interpolation between pollutant concentration breakpoints and corresponding AQI breakpoints. Overall AQI is the maximum valid pollutant sub-index.

| Pollutant | Averaging period | Canonical unit | Concentration breakpoints |
|---|---:|---|---|
| PM2.5 | 24 h | µg/m³ | 0, 30, 60, 90, 120, 250 |
| PM10 | 24 h | µg/m³ | 0, 50, 100, 250, 350, 430 |
| NO2 | 24 h | µg/m³ | 0, 40, 80, 180, 280, 400 |
| O3 | 8 h | µg/m³ | 0, 50, 100, 168, 208, 748 |
| CO | 8 h | mg/m³ | 0, 1, 2, 10, 17, 34 |
| SO2 | 24 h | µg/m³ | 0, 40, 80, 380, 800, 1600 |
| NH3 | 24 h | µg/m³ | 0, 200, 400, 800, 1200, 1800 |
| Pb | 24 h | µg/m³ | 0, 0.5, 1, 2, 3, 3.5 |

AQI breakpoint bands are 0, 50, 100, 200, 300, 400, 500. Each sub-index is linearly interpolated within its concentration band, rounded to an integer, and capped at 500.

For 24-hour pollutants, the calculation uses the rolling 24-hour window and requires at least 16 distinct hourly buckets, matching CPCB's published minimum. For CO and O3, the pipeline uses a complete eight-consecutive-hour window; this is intentionally conservative because CPCB's public methodology summary identifies an 8-hour average but does not state a lower minimum-hour count for those 8-hour windows.

Overall station-hour AQI is emitted only when at least three pollutant sub-indices are available and at least one is PM2.5 or PM10. Individual pollutant sub-indices are retained even when overall AQI is not eligible.

## Run

After the manual one-month ingestion and independent pilot audit:

```bash
python scripts/build_xkdr_cpcb_aqi.py \
  --data-dir data/xkdr_cpcb \
  --output-dir data/xkdr_cpcb_derived
```

Outputs:
- `xkdr_cpcb_normalized_hourly.parquet`: supported pollutants, canonical units, station-hour buckets.
- `xkdr_cpcb_reconstructed_aqi.parquet`: pollutant averaging windows and sub-indices.
- `xkdr_cpcb_station_hour_aqi.parquet`: station-hour pollutant sub-indices, dominant pollutant, eligibility and overall reconstructed AQI.
- `xkdr_cpcb_aqi_report.json`: method, source, counts and limitations.

The GitHub Actions manual pilot workflow runs the derivation only after ingestion and the independent raw-data audit pass, then uploads the derived outputs as a separate artifact.

## Validation gates still required

This pipeline is a deterministic method implementation, not yet evidence that the source observations are equivalent to CPCB's validated instrument feed. Before any historical training:
1. Compare reconstructed station-hour AQI and pollutant sub-indices against time-aligned official CPCB values for the same stations and periods.
2. Quantify agreement, bias, category confusion, and coverage by pollutant/station; investigate systematic discrepancies.
3. Review station metadata and sensor calibration/quality flags where available.
4. Expand to multiple months and years, verify source completeness and station identity continuity, and avoid leakage in chronological and station-held-out splits.
5. Only then create a separately versioned model-training dataset. Never replace the production dataset/model merely because this derivation ran successfully.

## References

- CPCB, *How is AQI calculated?*: https://www.cpcb.nic.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvSG93X0FRSV9DYWxjdWxhdGVkLnBkZg%3D%3D
- CPCB, *National Air Quality Index – Final Report* (CUPS/82/2014-15): https://app.cpcbccr.com/ccr_docs/FINAL-REPORT_AQI_.pdf
- Press Information Bureau, *National Air Quality Index launched*: https://www.pib.gov.in/newsite/PrintRelease.aspx?lang=2&reg=48&relid=110654
- XKDR Forum, *India Air Quality Database*: https://airquality.xkdr.org/#docs

Attribution: **XKDR Forum, India Air Quality Database; Central Pollution Control Board (CPCB) CAAQM network**. XKDR data is licensed under CC BY 4.0.


## Non-destructive concentration-tail diagnostics

The pilot workflow also runs `scripts/audit_xkdr_value_diagnostics.py`. It reports, by pollutant and source unit, the number of observations above the highest CPCB AQI concentration breakpoint, the number at least twice that breakpoint, the 99th and 99.9th percentiles, maxima, and representative station/time records. It also counts station/pollutant/hour buckets containing conflicting source values.

These are review queues, not automatic rejection rules. AQI breakpoints are index-mapping thresholds, not instrument-validity limits. A value above the top breakpoint can legitimately map to the capped sub-index of 500; it must not be deleted merely for exceeding that breakpoint. No raw observations are changed, dropped, winsorized, imputed, or relabelled.

The diagnostic report does not prove calibration or regulatory validity. Such a determination requires station-specific instrument QA/calibration metadata and preferably confirmation against the originating monitoring network.

## Official-source comparison status

The January 2024 pilot is **not yet cross-validated against time-aligned official CPCB station-hour AQI values**. CPCB's public methodology describes rolling 24-hour averages (8-hour for CO/O3), minimum 16-hour data for a 24-hour sub-index, and at least three pollutants including PM2.5 or PM10 for an overall AQI. The public dashboard exposes current network/AQI information, but a reproducible January 2024 station-hour export with stable station identifiers has not yet been obtained in this workflow.

Therefore:
- Do not label the XKDR-derived records as official CPCB AQI.
- Do not use this one-month derived dataset for production training or model replacement.
- Treat exact official-source agreement as an outstanding release gate, not as a test that has passed.
