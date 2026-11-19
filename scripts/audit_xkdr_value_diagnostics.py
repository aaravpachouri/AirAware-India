from __future__ import annotations

"""Non-destructive, unit-aware diagnostics for extreme CPCB source readings.

This report flags records for review; it never drops, edits, or reclassifies data.
A concentration above an AQI breakpoint is not automatically invalid: CPCB AQI
caps the corresponding sub-index at 500, while instrument validity needs source
QA/calibration evidence.
"""
import argparse
import json
from pathlib import Path

import duckdb

DEFAULT_DATA = Path("data/xkdr_cpcb")
TOP_BREAKPOINTS = {
    ("PM2.5", "µg/m³"): 250.0, ("PM10", "µg/m³"): 430.0,
    ("NO2", "µg/m³"): 400.0, ("Ozone", "µg/m³"): 748.0,
    ("CO", "mg/m³"): 34.0, ("SO2", "µg/m³"): 1600.0,
    ("NH3", "µg/m³"): 1800.0, ("Pb", "µg/m³"): 3.5,
}


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def audit(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "ingestion_manifest.json").read_text(encoding="utf-8"))
    files = [data_dir / item["file"] for item in manifest.get("partitions", [])]
    if not files or any(not path.is_file() for path in files):
        raise ValueError("Manifest is missing one or more source Parquet partitions.")
    paths = "[" + ",".join(_sql_string(str(path.resolve())) for path in files) + "]"
    conn = duckdb.connect(":memory:")
    columns = {row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM read_parquet({paths})").fetchall()}
    required = {"station_id", "parameter_name", "source_unit", "collected_at_ist", "value", "source"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Parquet schema missing required columns: {missing}")

    per_parameter = []
    extreme_examples = []
    for (parameter, unit), top in TOP_BREAKPOINTS.items():
        row = conn.execute(f"""
            SELECT count(*) AS rows,
                   count(*) FILTER (WHERE value > {top}) AS above_top_aqi_breakpoint,
                   count(*) FILTER (WHERE value >= {top} * 2) AS at_least_2x_top_breakpoint,
                   min(value), quantile_cont(value, 0.99),
                   quantile_cont(value, 0.999), max(value),
                   count(DISTINCT station_id)
            FROM read_parquet({paths})
            WHERE source = 'cpcb_caaqm'
              AND parameter_name = {_sql_string(parameter)}
              AND source_unit = {_sql_string(unit)}
        """).fetchone()
        per_parameter.append({
            "parameter": parameter, "source_unit": unit, "top_aqi_breakpoint": top,
            "rows": row[0], "above_top_aqi_breakpoint": row[1],
            "at_least_2x_top_breakpoint": row[2], "minimum": row[3],
            "p99": row[4], "p999": row[5], "maximum": row[6], "stations": row[7],
            "interpretation": "Review concentration tails; exceeding an AQI breakpoint is not itself proof of invalid measurement.",
        })
        examples = conn.execute(f"""
            SELECT station_id, station_name, city_name, state_name,
                   collected_at_ist, value, quality_status
            FROM read_parquet({paths})
            WHERE source = 'cpcb_caaqm'
              AND parameter_name = {_sql_string(parameter)}
              AND source_unit = {_sql_string(unit)}
              AND value >= {top} * 2
            ORDER BY value DESC
            LIMIT 10
        """).fetchall()
        for item in examples:
            extreme_examples.append({
                "parameter": parameter, "unit": unit, "station_id": item[0],
                "station_name": item[1], "city": item[2], "state": item[3],
                "collected_at_ist": str(item[4]), "value": item[5],
                "quality_status": item[6],
            })

    conflicts = conn.execute(f"""
        SELECT count(*) FROM (
          SELECT station_id, parameter_name, source_unit, collected_at_ist
          FROM read_parquet({paths})
          WHERE source = 'cpcb_caaqm'
          GROUP BY 1,2,3,4
          HAVING count(*) > 1 AND count(DISTINCT value) > 1
        )
    """).fetchone()[0]
    conn.close()
    return {
        "schema_version": 1, "status": "diagnostics_complete",
        "source_months": [item.get("month") for item in manifest.get("partitions", [])],
        "source_dataset": manifest.get("dataset"),
        "source_license": manifest.get("source_license"),
        "required_attribution": manifest.get("required_attribution"),
        "raw_observations_modified": False,
        "top_breakpoint_diagnostics": per_parameter,
        "station_parameter_hour_buckets_with_conflicting_source_values": conflicts,
        "extreme_examples_at_least_2x_top_breakpoint": extreme_examples,
        "interpretation_limits": [
            "AQI breakpoints are index mapping thresholds, not instrument-validity limits.",
            "Values above the top concentration breakpoint map to the capped AQI sub-index; do not delete them solely for exceeding it.",
            "Values at or above twice the top breakpoint are surfaced for manual source/instrument review only.",
            "No record is removed, winsorized, imputed, or relabelled by this diagnostic.",
            "A scientific validity determination requires monitor calibration, QA flags, and preferably source-network confirmation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("reports/xkdr_value_diagnostics.json"))
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "raw_observations_modified": report["raw_observations_modified"],
        "review_examples": len(report["extreme_examples_at_least_2x_top_breakpoint"]),
        "conflicting_hour_buckets": report["station_parameter_hour_buckets_with_conflicting_source_values"],
        "report": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
