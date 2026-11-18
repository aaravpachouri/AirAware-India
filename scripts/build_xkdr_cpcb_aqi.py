from __future__ import annotations

"""Create an isolated, CPCB-methodology-derived dataset from XKDR CPCB Parquet.

Raw partitions are read-only. This pipeline writes normalized hourly observations
and reconstructed AQI to a separate output directory; it never trains a model.
"""
import argparse
import json
import math
from pathlib import Path

import duckdb

# CPCB National AQI pollutant set and concentration breakpoints. Concentration
# units are µg/m³ except CO (mg/m³); index bands are 0-50, 51-100, 101-200,
# 201-300, 301-400, 401-500. Upper-band values are capped at 500.
# Values follow CPCB's National AQI technical report / official breakpoint table.
BREAKPOINTS = {
    "pm25": (("ug/m3", (0, 30, 60, 90, 120, 250)),),
    "pm10": (("ug/m3", (0, 50, 100, 250, 350, 430)),),
    "no2": (("ug/m3", (0, 40, 80, 180, 280, 400)),),
    "o3": (("ug/m3", (0, 50, 100, 168, 208, 748)),),
    "co": (("mg/m3", (0, 1, 2, 10, 17, 34)),),
    "so2": (("ug/m3", (0, 40, 80, 380, 800, 1600)),),
    "nh3": (("ug/m3", (0, 200, 400, 800, 1200, 1800)),),
    "pb": (("ug/m3", (0, 0.5, 1, 2, 3, 3.5)),),
}
INDEX_BREAKPOINTS = (0, 50, 100, 200, 300, 400, 500)
PARAMETER_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "Ozone": "o3",
    "CO": "co",
    "SO2": "so2",
    "NH3": "nh3",
    "Pb": "pb",
}
UNIT_ALIASES = {
    "µg/m³": "ug/m3",
    "μg/m³": "ug/m3",
    "ug/m3": "ug/m3",
    "µg/m3": "ug/m3",
    "μg/m3": "ug/m3",
    "mg/m³": "mg/m3",
    "mg/m3": "mg/m3",
}
AVERAGING = {"pm25": 24, "pm10": 24, "no2": 24, "so2": 24, "nh3": 24, "pb": 24, "co": 8, "o3": 8}
MIN_24H_VALID_HOURS = 16
MIN_8H_CONSECUTIVE_HOURS = 8


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _subindex_sql(pollutant: str, concentration_sql: str) -> str:
    points = BREAKPOINTS[pollutant][0][1]
    clauses = []
    for idx in range(len(points) - 1):
        low, high = points[idx], points[idx + 1]
        ilo, ihi = INDEX_BREAKPOINTS[idx], INDEX_BREAKPOINTS[idx + 1]
        clauses.append(
            f"WHEN {concentration_sql} <= {high} THEN "
            f"{ilo} + ({concentration_sql} - {low}) * "
            f"({ihi} - {ilo}) / ({high} - {low})"
        )
    return "CASE WHEN " + concentration_sql + " < 0 THEN NULL " + " ".join(clauses) + " ELSE 500 END"


def build(data_dir: Path, output_dir: Path) -> dict:
    manifest_path = data_dir / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition_files = [data_dir / item["file"] for item in manifest.get("partitions", [])]
    if not partition_files or any(not path.is_file() for path in partition_files):
        raise ValueError("Manifest is missing one or more source Parquet partitions.")
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "xkdr_cpcb_normalized_hourly.parquet"
    aqi_path = output_dir / "xkdr_cpcb_reconstructed_aqi.parquet"
    report_path = output_dir / "xkdr_cpcb_aqi_report.json"

    paths = "[" + ",".join(_sql_string(str(p.resolve())) for p in partition_files) + "]"
    conn = duckdb.connect()
    conn.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet({paths})")

    # Only exact, documented unit spellings are canonicalized. No ppb/ppm to
    # mass-concentration conversion is attempted because temperature/pressure
    # metadata are absent. NO, NOx, VOCs and other non-NAQI parameters remain
    # in the raw source only and are not silently substituted for NAQI inputs.
    param_case = "CASE parameter_name " + " ".join(
        f"WHEN {_sql_string(source)} THEN {_sql_string(target)}"
        for source, target in PARAMETER_MAP.items()
    ) + " ELSE NULL END"
    unit_case = "CASE source_unit " + " ".join(
        f"WHEN {_sql_string(source)} THEN {_sql_string(target)}"
        for source, target in UNIT_ALIASES.items()
    ) + " ELSE NULL END"
    normalized_unit = f"({unit_case})"
    compatible = " OR ".join(
        f"(pollutant = {_sql_string(p)} AND canonical_unit = {_sql_string(u)})"
        for p, entries in BREAKPOINTS.items() for u, _ in entries
    )
    conn.execute(f"""
      COPY (
        WITH mapped AS (
          SELECT station_id, station_name, state_name, city_name,
                 collected_at_ist, collected_at_utc, value, source_unit,
                 {param_case} AS pollutant,
                 {normalized_unit} AS canonical_unit
          FROM raw
          WHERE source = 'cpcb_caaqm'
        ),
        accepted AS (
          SELECT *, date_trunc('hour', collected_at_ist) AS hour_ist
          FROM mapped
          WHERE pollutant IS NOT NULL
            AND canonical_unit IS NOT NULL
            AND ({compatible})
            AND value IS NOT NULL AND isfinite(value) AND value >= 0
            AND collected_at_ist IS NOT NULL
        ),
        hourly AS (
          SELECT station_id, any_value(station_name) AS station_name,
                 any_value(state_name) AS state_name,
                 any_value(city_name) AS city_name,
                 pollutant, canonical_unit, hour_ist,
                 (hour_ist AT TIME ZONE 'Asia/Kolkata') AS hour_utc,
                 avg(value) AS concentration,
                 count(*) AS source_observations
          FROM accepted
          GROUP BY station_id, pollutant, canonical_unit, hour_ist
        )
        SELECT * FROM hourly
      ) TO {_sql_string(str(normalized_path.resolve()))}
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Rolling windows are computed on deduplicated station/pollutant/hour
    # buckets. For 24h pollutants, CPCB's minimum of 16 hourly observations is
    # required. For CO/O3, a complete eight-consecutive-hour window is required.
    # The 8-hour rule is deliberately conservative where CPCB's public summary
    # does not specify a lower minimum for the 8-hour pollutant window.
    subindex_cases = " ".join(
        f"WHEN pollutant = {_sql_string(p)} THEN {_subindex_sql(p, 'averaged_concentration')}"
        for p in BREAKPOINTS
    )
    conn.execute(f"""
      COPY (
        WITH windows AS (
          SELECT *,
            avg(concentration) OVER (
              PARTITION BY station_id, pollutant
              ORDER BY hour_ist RANGE BETWEEN INTERVAL 23 HOURS PRECEDING AND CURRENT ROW
            ) AS avg_24h,
            count(*) OVER (
              PARTITION BY station_id, pollutant
              ORDER BY hour_ist RANGE BETWEEN INTERVAL 23 HOURS PRECEDING AND CURRENT ROW
            ) AS count_24h,
            avg(concentration) OVER (
              PARTITION BY station_id, pollutant
              ORDER BY hour_ist RANGE BETWEEN INTERVAL 7 HOURS PRECEDING AND CURRENT ROW
            ) AS avg_8h,
            count(*) OVER (
              PARTITION BY station_id, pollutant
              ORDER BY hour_ist RANGE BETWEEN INTERVAL 7 HOURS PRECEDING AND CURRENT ROW
            ) AS count_8h
          FROM read_parquet({_sql_string(str(normalized_path.resolve()))})
        ),
        averaged AS (
          SELECT *,
            CASE
              WHEN pollutant IN ('co','o3') AND count_8h = {MIN_8H_CONSECUTIVE_HOURS}
                THEN avg_8h
              WHEN pollutant NOT IN ('co','o3') AND count_24h >= {MIN_24H_VALID_HOURS}
                THEN avg_24h
              ELSE NULL
            END AS averaged_concentration,
            CASE WHEN pollutant IN ('co','o3') THEN '8h' ELSE '24h' END AS averaging_period
          FROM windows
        ),
        indexed AS (
          SELECT *,
            CASE {subindex_cases} ELSE NULL END AS raw_sub_index
          FROM averaged
          WHERE averaged_concentration IS NOT NULL
        )
        SELECT station_id, station_name, state_name, city_name, hour_ist, hour_utc,
               pollutant, canonical_unit, averaging_period,
               averaged_concentration,
               least(500, greatest(0, round(raw_sub_index)))::INTEGER AS sub_index
        FROM indexed
        WHERE raw_sub_index IS NOT NULL
      ) TO {_sql_string(str(aqi_path.resolve()))}
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    pollutant_counts = conn.execute(f"""
      SELECT pollutant, canonical_unit, count(*) AS hourly_rows,
             count(DISTINCT station_id) AS stations
      FROM read_parquet({_sql_string(str(normalized_path.resolve()))})
      GROUP BY 1,2 ORDER BY 1,2
    """).fetchall()
    derived_count = conn.execute(f"SELECT count(*) FROM read_parquet({_sql_string(str(aqi_path.resolve()))})").fetchone()[0]
    # Reconstruct station-hour AQI from valid pollutant sub-indices. Three
    # valid pollutants are required, including PM2.5 or PM10.
    aqi_final = output_dir / "xkdr_cpcb_station_hour_aqi.parquet"
    conn.execute(f"""
      COPY (
        WITH pivoted AS (
          SELECT station_id, any_value(station_name) AS station_name,
                 any_value(state_name) AS state_name,
                 any_value(city_name) AS city_name, hour_ist, hour_utc,
                 count(*) AS valid_pollutants,
                 max(sub_index) AS candidate_aqi,
                 first(pollutant ORDER BY sub_index DESC, pollutant) AS dominant_pollutant,
                 max(sub_index) FILTER (WHERE pollutant='pm25') AS subindex_pm25,
                 max(sub_index) FILTER (WHERE pollutant='pm10') AS subindex_pm10,
                 max(sub_index) FILTER (WHERE pollutant='no2') AS subindex_no2,
                 max(sub_index) FILTER (WHERE pollutant='so2') AS subindex_so2,
                 max(sub_index) FILTER (WHERE pollutant='nh3') AS subindex_nh3,
                 max(sub_index) FILTER (WHERE pollutant='co') AS subindex_co,
                 max(sub_index) FILTER (WHERE pollutant='o3') AS subindex_o3,
                 max(sub_index) FILTER (WHERE pollutant='pb') AS subindex_pb
          FROM read_parquet({_sql_string(str(aqi_path.resolve()))})
          GROUP BY station_id, hour_ist, hour_utc
        )
        SELECT *, (valid_pollutants >= 3 AND
                   (subindex_pm25 IS NOT NULL OR subindex_pm10 IS NOT NULL)) AS aqi_eligible,
               CASE WHEN valid_pollutants >= 3 AND
                   (subindex_pm25 IS NOT NULL OR subindex_pm10 IS NOT NULL)
                 THEN candidate_aqi ELSE NULL END AS reconstructed_cpcb_aqi
        FROM pivoted
      ) TO {_sql_string(str(aqi_final.resolve()))}
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    overall = conn.execute(f"""
      SELECT count(*) AS station_hours,
             count(*) FILTER (WHERE aqi_eligible) AS eligible_station_hours,
             count(DISTINCT station_id) AS stations
      FROM read_parquet({_sql_string(str(aqi_final.resolve()))})
    """).fetchone()
    conn.close()

    report = {
        "schema_version": 1,
        "status": "derived_dataset_created",
        "source_dataset": manifest.get("dataset"),
        "source_license": manifest.get("source_license"),
        "required_attribution": manifest.get("required_attribution"),
        "source_months": [p.get("month") for p in manifest.get("partitions", [])],
        "raw_observations_are_unchanged": True,
        "unit_policy": {
            "mass_concentration_aliases_normalized": UNIT_ALIASES,
            "canonical_units": {"all_NAQI_pollutants_except_CO": "ug/m3", "CO": "mg/m3"},
            "ppb_ppm_conversion": "not attempted; temperature and pressure metadata are unavailable",
            "unsupported_and_non_NAQI_parameters": "retained in raw source, excluded from this derived AQI dataset",
        },
        "aqi_methodology": {
            "breakpoints": {p: {"unit": e[0][0], "concentrations": list(e[0][1])} for p,e in BREAKPOINTS.items()},
            "index_breakpoints": list(INDEX_BREAKPOINTS),
            "averaging_periods_hours": AVERAGING,
            "minimum_24h_valid_hourly_observations": MIN_24H_VALID_HOURS,
            "minimum_8h_consecutive_hourly_observations": MIN_8H_CONSECUTIVE_HOURS,
            "overall_aqi_minimum_pollutants": 3,
            "overall_aqi_requires_pm25_or_pm10": True,
            "overall_aqi_is_maximum_valid_pollutant_subindex": True,
            "upper_aqi_cap": 500,
        },
        "normalized_pollutant_hourly_rows": [
            {"pollutant": r[0], "unit": r[1], "rows": r[2], "stations": r[3]}
            for r in pollutant_counts
        ],
        "pollutant_subindex_rows": derived_count,
        "station_hour_aqi": {
            "rows": overall[0], "eligible_rows": overall[1], "stations": overall[2],
            "output_file": aqi_final.name,
        },
        "outputs": [normalized_path.name, aqi_path.name, aqi_final.name, report_path.name],
        "limitations": [
            "Reconstructed AQI is a research derivation from XKDR-republished observations, not an official CPCB-published AQI.",
            "XKDR states readings are published as received and are not independently validated by XKDR.",
            "Lead is part of the CPCB eight-pollutant index but is absent from the January 2024 pilot parameter set.",
            "NO, NOx, benzene and other VOCs are not substituted for the eight CPCB AQI pollutants.",
            "This script does not train, evaluate, or update any production forecast model.",
            "The 8-hour CO/O3 requirement uses a complete 8-hour window as a conservative implementation choice.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xkdr_cpcb"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/xkdr_cpcb_derived"))
    args = parser.parse_args()
    report = build(args.data_dir, args.output_dir)
    print(json.dumps({
        "status": report["status"],
        "normalized_hourly_rows": sum(x["rows"] for x in report["normalized_pollutant_hourly_rows"]),
        "pollutant_subindex_rows": report["pollutant_subindex_rows"],
        "eligible_station_hours": report["station_hour_aqi"]["eligible_rows"],
        "stations": report["station_hour_aqi"]["stations"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
