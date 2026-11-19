import json
from pathlib import Path

import duckdb

from scripts.audit_xkdr_value_diagnostics import audit


def _fixture(data: Path, values):
    data.mkdir()
    parquet = data / "measurements.parquet"
    conn = duckdb.connect()
    rows = ",\n".join(
        "('s1','Station 1','City','State','PM2.5','µg/m³',TIMESTAMP '2024-01-01 00:00:00',"
        + str(value) + ",'cpcb_caaqm','eligible_raw_observation')"
        for value in values
    )
    conn.execute(f"""
      COPY (
        SELECT * FROM (VALUES {rows})
        AS t(station_id,station_name,city_name,state_name,parameter_name,source_unit,
             collected_at_ist,value,source,quality_status)
      ) TO '{parquet}' (FORMAT PARQUET)
    """)
    conn.close()
    (data / "ingestion_manifest.json").write_text(json.dumps({
        "dataset": "fixture", "source_license": "CC BY 4.0",
        "required_attribution": "fixture",
        "partitions": [{"month": "2024-01", "file": "measurements.parquet"}],
    }))


def test_extreme_values_are_flagged_without_mutating_source(tmp_path: Path):
    data = tmp_path / "data"
    _fixture(data, [100.0, 600.0])
    report = audit(data)
    pm25 = next(x for x in report["top_breakpoint_diagnostics"] if x["parameter"] == "PM2.5")
    assert pm25["above_top_aqi_breakpoint"] == 1
    assert pm25["at_least_2x_top_breakpoint"] == 1
    assert len(report["extreme_examples_at_least_2x_top_breakpoint"]) == 1
    assert report["raw_observations_modified"] is False


def test_conflicting_values_in_same_station_pollutant_hour_are_reported(tmp_path: Path):
    data = tmp_path / "data"
    _fixture(data, [100.0, 110.0])
    report = audit(data)
    assert report["station_parameter_hour_buckets_with_conflicting_source_values"] == 1
