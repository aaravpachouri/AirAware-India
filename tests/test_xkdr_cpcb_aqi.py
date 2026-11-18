from __future__ import annotations

import json
from pathlib import Path

import duckdb

from scripts.build_xkdr_cpcb_aqi import BREAKPOINTS, _subindex_sql, build


def _write_fixture(root: Path, *, hours: int = 24) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "pilot.parquet"
    conn = duckdb.connect()
    rows = []
    for hour in range(hours):
        # All timestamps are naive IST, matching the XKDR source contract.
        stamp = f"2024-01-01 {hour:02d}:00:00"
        rows.extend([
            ("site_1", "Test Station", "Delhi", "Delhi", "PM2.5", "µg/m³", stamp, 60.0, "cpcb_caaqm"),
            ("site_1", "Test Station", "Delhi", "Delhi", "PM10", "µg/m³", stamp, 100.0, "cpcb_caaqm"),
            ("site_1", "Test Station", "Delhi", "Delhi", "NO2", "µg/m³", stamp, 80.0, "cpcb_caaqm"),
            ("site_1", "Test Station", "Delhi", "Delhi", "CO", "mg/m³", stamp, 2.0, "cpcb_caaqm"),
            # NOx is not a National AQI pollutant; ppb is not converted.
            ("site_1", "Test Station", "Delhi", "Delhi", "NOx", "ppb", stamp, 40.0, "cpcb_caaqm"),
        ])
    conn.execute("""
      CREATE TABLE observations(
        station_id VARCHAR, station_name VARCHAR, state_name VARCHAR, city_name VARCHAR,
        parameter_name VARCHAR, source_unit VARCHAR, collected_at_ist TIMESTAMP,
        value DOUBLE, source VARCHAR
      )
    """)
    conn.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.execute("ALTER TABLE observations ADD COLUMN collected_at_utc TIMESTAMPTZ")
    conn.execute("UPDATE observations SET collected_at_utc = collected_at_ist AT TIME ZONE 'Asia/Kolkata'")
    conn.execute(f"COPY observations TO '{parquet.as_posix()}' (FORMAT PARQUET)")
    conn.close()
    (root / "ingestion_manifest.json").write_text(json.dumps({
        "dataset": "fixture", "source_license": "CC BY 4.0",
        "required_attribution": "XKDR Forum",
        "partitions": [{"file": parquet.name, "month": "2024-01"}],
    }), encoding="utf-8")
    return parquet


def test_cpcb_breakpoint_interpolation_and_cap():
    conn = duckdb.connect()
    expr = _subindex_sql("pm25", "x")
    assert conn.execute(f"SELECT {expr} FROM (SELECT 30.0 AS x)").fetchone()[0] == 50
    assert conn.execute(f"SELECT {expr} FROM (SELECT 60.0 AS x)").fetchone()[0] == 100
    assert conn.execute(f"SELECT {expr} FROM (SELECT 90.0 AS x)").fetchone()[0] == 200
    assert conn.execute(f"SELECT {expr} FROM (SELECT 120.0 AS x)").fetchone()[0] == 300
    assert conn.execute(f"SELECT {expr} FROM (SELECT 250.0 AS x)").fetchone()[0] == 400
    assert conn.execute(f"SELECT {expr} FROM (SELECT 251.0 AS x)").fetchone()[0] == 500
    conn.close()


def test_all_eight_cpcb_breakpoint_tables_have_expected_units_and_periods():
    assert BREAKPOINTS["pm25"][0] == ("ug/m3", (0, 30, 60, 90, 120, 250))
    assert BREAKPOINTS["co"][0] == ("mg/m3", (0, 1, 2, 10, 17, 34))
    assert set(BREAKPOINTS) == {"pm25", "pm10", "no2", "o3", "co", "so2", "nh3", "pb"}


def test_pipeline_normalizes_units_and_reconstructs_station_hour_aqi(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    _write_fixture(source)
    report = build(source, output)

    assert report["status"] == "derived_dataset_created"
    assert report["station_hour_aqi"]["eligible_rows"] == 9
    assert report["station_hour_aqi"]["stations"] == 1

    conn = duckdb.connect()
    aqi_file = output / "xkdr_cpcb_station_hour_aqi.parquet"
    row = conn.execute(f"""
      SELECT aqi_eligible, reconstructed_cpcb_aqi, dominant_pollutant,
             subindex_pm25, subindex_pm10, subindex_no2
      FROM read_parquet('{aqi_file.as_posix()}')
      ORDER BY hour_ist DESC LIMIT 1
    """).fetchone()
    assert row == (True, 100, "no2", 100, 100, 100)

    normalized_file = output / "xkdr_cpcb_normalized_hourly.parquet"
    units = conn.execute(f"""
      SELECT pollutant, canonical_unit, count(*)
      FROM read_parquet('{normalized_file.as_posix()}')
      GROUP BY 1,2 ORDER BY 1
    """).fetchall()
    assert ("co", "mg/m3", 24) in units
    assert ("pm25", "ug/m3", 24) in units
    assert all(p != "nox" for p, _, _ in units)
    conn.close()


def test_24h_subindex_is_not_emitted_with_fewer_than_16_hours(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    _write_fixture(source, hours=15)
    report = build(source, output)
    assert report["pollutant_subindex_rows"] == 8
    assert report["station_hour_aqi"]["eligible_rows"] == 0


def test_pipeline_does_not_attempt_ppb_conversion_or_modify_raw_source(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    parquet = _write_fixture(source)
    before = parquet.read_bytes()
    build(source, output)
    assert parquet.read_bytes() == before
    conn = duckdb.connect()
    raw_nox = conn.execute(
        f"SELECT count(*) FROM read_parquet('{parquet.as_posix()}') WHERE parameter_name='NOx'"
    ).fetchone()[0]
    derived_nox = conn.execute(
        f"SELECT count(*) FROM read_parquet('{(output / 'xkdr_cpcb_normalized_hourly.parquet').as_posix()}') WHERE pollutant='nox'"
    ).fetchone()[0]
    assert raw_nox == 15
    assert derived_nox == 0
    conn.close()
