from __future__ import annotations

"""Independent QA report for an isolated XKDR CPCB Parquet pilot.

This report is diagnostic only: it never edits observations, converts units,
relabels records as regulator-valid, or creates a model-training dataset.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import duckdb

DEFAULT_DATA = Path("data/xkdr_cpcb")
REQUIRED_COLUMNS = {
    "station_id", "station_name", "state_name", "city_name", "parameter_name",
    "source_unit", "collected_at_ist", "collected_at_utc", "value", "source",
    "quality_status",
}
# Broad plausibility screens are warnings only, not exclusion criteria.
PLAUSIBILITY_RANGES = {
    "PM2.5": (0, 2000), "PM10": (0, 3000), "NO": (0, 5000),
    "NO2": (0, 5000), "NOx": (0, 10000), "NH3": (0, 10000),
    "SO2": (0, 10000), "CO": (0, 100000), "Ozone": (0, 5000),
    "Benzene": (0, 10000), "Toluene": (0, 20000), "Xylene": (0, 20000),
    "O_Xylene": (0, 20000), "MP-Xylene": (0, 20000), "Eth-Benzene": (0, 20000),
}


def _rows(conn, query: str):
    return [dict(zip([c[0] for c in conn.description], row)) for row in conn.execute(query).fetchall()]


def audit(data_dir: Path) -> dict:
    manifest_path = data_dir / "ingestion_manifest.json"
    registry_path = data_dir / "cpcb_station_registry.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = registry_payload.get("data") if isinstance(registry_payload, dict) else registry_payload
    if not isinstance(registry, list):
        raise ValueError("Station registry must be a list or an object containing a data list.")

    files = [data_dir / item["file"] for item in manifest.get("partitions", [])]
    if not files or any(not p.is_file() for p in files):
        raise ValueError("One or more manifest Parquet partition files are missing.")
    conn = duckdb.connect(":memory:")
    paths_sql = "[" + ",".join("'" + str(p.resolve()).replace("'", "''") + "'" for p in files) + "]"
    schema = _rows(conn, f"DESCRIBE SELECT * FROM read_parquet({paths_sql})")
    columns = {row["column_name"] for row in schema}
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        raise ValueError(f"Parquet schema missing required columns: {missing_columns}")

    total = conn.execute(f"SELECT count(*) FROM read_parquet({paths_sql})").fetchone()[0]
    overview = conn.execute(f"""
      SELECT
        count(*) FILTER (WHERE source <> 'cpcb_caaqm' OR source IS NULL) AS non_cpcb_rows,
        count(*) FILTER (WHERE station_id IS NULL OR trim(station_id)='') AS missing_station_id,
        count(*) FILTER (WHERE parameter_name IS NULL OR trim(parameter_name)='') AS missing_parameter,
        count(*) FILTER (WHERE source_unit IS NULL OR trim(source_unit)='') AS missing_unit,
        count(*) FILTER (WHERE collected_at_ist IS NULL) AS missing_ist_timestamp,
        count(*) FILTER (WHERE collected_at_utc IS NULL) AS missing_utc_timestamp,
        count(*) FILTER (WHERE value IS NULL OR NOT isfinite(value)) AS invalid_numeric,
        count(*) FILTER (WHERE value < 0) AS negative_values,
        count(*) FILTER (WHERE quality_status='eligible_raw_observation') AS ingestion_eligible_rows,
        count(DISTINCT station_id) AS distinct_stations,
        count(DISTINCT parameter_name) AS distinct_parameters,
        min(collected_at_ist) AS min_ist,
        max(collected_at_ist) AS max_ist
      FROM read_parquet({paths_sql})
    """).fetchone()
    names = [c[0] for c in conn.description]
    overview = dict(zip(names, overview))

    unit_distribution = _rows(conn, f"""
      SELECT parameter_name, source_unit, count(*) AS rows,
             count(*) FILTER (WHERE quality_status='eligible_raw_observation') AS ingestion_eligible_rows
      FROM read_parquet({paths_sql})
      GROUP BY 1,2 ORDER BY 1,2
    """)
    parameter_distribution = _rows(conn, f"""
      SELECT parameter_name, count(*) AS rows,
             count(DISTINCT station_id) AS stations,
             min(value) AS min_value, quantile_cont(value, 0.01) AS p01,
             quantile_cont(value, 0.50) AS median,
             quantile_cont(value, 0.99) AS p99, max(value) AS max_value
      FROM read_parquet({paths_sql})
      GROUP BY 1 ORDER BY 1
    """)
    quality_distribution = _rows(conn, f"""
      SELECT quality_status, count(*) AS rows
      FROM read_parquet({paths_sql})
      GROUP BY 1 ORDER BY 1
    """)
    duplicates = conn.execute(f"""
      SELECT count(*) FROM (
        SELECT station_id, parameter_name, collected_at_ist, source_unit, value, count(*) AS n
        FROM read_parquet({paths_sql})
        GROUP BY ALL HAVING count(*) > 1
      )
    """).fetchone()[0]
    duplicate_excess_rows = conn.execute(f"""
      SELECT coalesce(sum(n-1),0) FROM (
        SELECT count(*) AS n FROM read_parquet({paths_sql})
        GROUP BY station_id, parameter_name, collected_at_ist, source_unit, value
        HAVING count(*) > 1
      )
    """).fetchone()[0]
    timestamp_check = conn.execute(f"""
      SELECT count(*) AS checked_rows,
             count(*) FILTER (
               WHERE abs(epoch(collected_at_utc) - epoch(timezone('Asia/Kolkata', collected_at_ist))) > 1
             ) AS ist_utc_mismatch_rows
      FROM read_parquet({paths_sql})
      WHERE collected_at_ist IS NOT NULL AND collected_at_utc IS NOT NULL
    """).fetchone()
    conn.close()

    registry_ids = {str(row.get("station_id")) for row in registry if row.get("station_id")}
    registry_missing_ids = sum(1 for row in registry if not row.get("station_id"))
    registry_coords_invalid = []
    registry_coord_missing = 0
    for row in registry:
        lat, lon = row.get("latitude"), row.get("longitude")
        if lat is None or lon is None:
            registry_coord_missing += 1
        else:
            try:
                if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
                    registry_coords_invalid.append(str(row.get("station_id")))
            except (TypeError, ValueError):
                registry_coords_invalid.append(str(row.get("station_id")))
    observed_ids = set()
    # Station IDs are a small group-by result; read only this column.
    conn = duckdb.connect(":memory:")
    observed_ids = {str(row[0]) for row in conn.execute(
        f"SELECT DISTINCT station_id FROM read_parquet({paths_sql}) WHERE station_id IS NOT NULL"
    ).fetchall()}
    conn.close()

    outside_ranges = []
    for row in parameter_distribution:
        bounds = PLAUSIBILITY_RANGES.get(row["parameter_name"])
        if bounds and row["min_value"] is not None:
            if row["min_value"] < bounds[0] or row["max_value"] > bounds[1]:
                outside_ranges.append({
                    "parameter": row["parameter_name"], "unit": "mixed_or_source_labelled",
                    "min_value": row["min_value"], "max_value": row["max_value"],
                    "diagnostic_range": list(bounds),
                    "note": "Investigate; this broad screen does not invalidate or remove readings.",
                })

    registry_observed = observed_ids & registry_ids
    observed_not_in_registry = sorted(observed_ids - registry_ids)
    registry_not_observed = sorted(registry_ids - observed_ids)
    checks = {
        "all_rows_are_cpcb_source": overview["non_cpcb_rows"] == 0,
        "no_missing_station_ids": overview["missing_station_id"] == 0,
        "no_missing_timestamps": overview["missing_ist_timestamp"] == 0 and overview["missing_utc_timestamp"] == 0,
        "ist_utc_conversion_consistent": timestamp_check[1] == 0,
        "no_duplicate_dedupe_keys_remain": duplicates == 0,
        "all_observed_stations_have_registry_entry": len(observed_not_in_registry) == 0,
        "registry_coordinates_in_range": len(registry_coords_invalid) == 0,
    }
    report = {
        "schema_version": 1,
        "status": "audit_complete",
        "dataset": manifest.get("dataset"),
        "source_license": manifest.get("source_license"),
        "required_attribution": manifest.get("required_attribution"),
        "requested_start_month": manifest.get("requested_start_month"),
        "requested_end_month": manifest.get("requested_end_month"),
        "manifest_rows": manifest.get("total_rows"),
        "observed_rows": total,
        "row_count_matches_manifest": total == manifest.get("total_rows"),
        "schema_columns": sorted(columns),
        "overview": overview,
        "quality_status_distribution": quality_distribution,
        "parameter_distribution": parameter_distribution,
        "parameter_unit_distribution": unit_distribution,
        "timestamp_audit": {
            "checked_rows": timestamp_check[0],
            "ist_utc_mismatch_rows": timestamp_check[1],
            "policy": "collected_at_ist is a naive IST wall time; collected_at_utc is expected to represent the same instant",
        },
        "duplicate_audit": {
            "duplicate_key_groups_remaining": duplicates,
            "excess_rows_remaining": duplicate_excess_rows,
            "key": ["station_id", "parameter_name", "collected_at_ist", "source_unit", "value"],
            "limitation": "The ingestion output has already collapsed exact-key duplicates; this report cannot recover their pre-deduplication count.",
        },
        "station_reconciliation": {
            "registry_rows": len(registry),
            "registry_unique_ids": len(registry_ids),
            "registry_rows_missing_id": registry_missing_ids,
            "observed_station_ids": len(observed_ids),
            "observed_ids_in_registry": len(registry_observed),
            "observed_ids_not_in_registry": observed_not_in_registry,
            "registry_ids_not_observed_in_period": registry_not_observed,
            "registry_coordinates_missing": registry_coord_missing,
            "registry_coordinates_invalid": registry_coords_invalid,
        },
        "plausibility_diagnostics": {
            "out_of_broad_range_parameter_summaries": outside_ranges,
            "note": "Broad screening only. Thresholds are diagnostics, not unit-aware regulatory limits; do not use as automatic exclusions.",
        },
        "checks": checks,
        "limitations": [
            "An ingestion eligibility flag is not a scientific or regulatory validity determination.",
            "Mixed units must not be pooled or compared without pollutant-specific unit normalization.",
            "This audit does not reconstruct CPCB AQI, reconcile OpenAQ IDs, or train a model.",
            "The January pilot is one month and does not establish historical completeness or representativeness.",
        ],
    }
    report["overall_pass"] = all(checks.values()) and report["row_count_matches_manifest"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("reports/xkdr_pilot_quality_audit.json"))
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "overall_pass": report["overall_pass"],
        "observed_rows": report["observed_rows"],
        "observed_stations": report["overview"]["distinct_stations"],
        "failed_checks": [k for k, v in report["checks"].items() if not v],
        "report": str(args.output),
    }, indent=2))
    # QA warnings do not discard data. A failed hard consistency check blocks
    # downstream training and makes the workflow fail visibly.
    if not report["overall_pass"]:
        raise SystemExit("XKDR pilot QA failed; do not use this dataset for training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"XKDR pilot QA failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
