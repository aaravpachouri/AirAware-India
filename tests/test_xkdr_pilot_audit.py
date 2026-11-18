from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from scripts.audit_xkdr_pilot import audit


def make_fixture(root: Path, *, bad_utc: bool = False, unknown_station: bool = False):
    root.mkdir(parents=True)
    parquet = root / "cpcb_measurements_2024_01.parquet"
    conn = duckdb.connect()
    station_id = "site_1" if not unknown_station else "site_missing"
    utc_expr = "TIMESTAMPTZ '2024-01-01 00:00:00+00'" if bad_utc else "TIMESTAMPTZ '2023-12-31 18:30:00+00'"
    conn.execute(f"""
      COPY (
        SELECT
          '{station_id}'::VARCHAR AS station_id,
          'Test Station'::VARCHAR AS station_name,
          'Delhi'::VARCHAR AS state_name,
          'Delhi'::VARCHAR AS city_name,
          'PM2.5'::VARCHAR AS parameter_name,
          'µg/m³'::VARCHAR AS source_unit,
          TIMESTAMP '2024-01-01 00:00:00' AS collected_at_ist,
          {utc_expr} AS collected_at_utc,
          20.0::DOUBLE AS value,
          'cpcb_caaqm'::VARCHAR AS source,
          'eligible_raw_observation'::VARCHAR AS quality_status
      ) TO '{parquet.as_posix()}' (FORMAT PARQUET)
    """)
    conn.close()
    (root / "cpcb_station_registry.json").write_text(json.dumps({
        "meta": {"rows": 1}, "data": [{
            "station_id": "site_1", "latitude": 28.6, "longitude": 77.2
        }]
    }), encoding="utf-8")
    (root / "ingestion_manifest.json").write_text(json.dumps({
        "dataset": "test", "source_license": "CC BY 4.0",
        "required_attribution": "XKDR Forum",
        "requested_start_month": "2024-01", "requested_end_month": "2024-01",
        "total_rows": 1, "partitions": [{"file": parquet.name}]
    }), encoding="utf-8")


def test_audit_passes_consistent_single_station_pilot(tmp_path):
    data = tmp_path / "pilot"
    make_fixture(data)
    result = audit(data)
    assert result["overall_pass"] is True
    assert result["observed_rows"] == 1
    assert result["timestamp_audit"]["ist_utc_mismatch_rows"] == 0
    assert result["station_reconciliation"]["observed_ids_not_in_registry"] == []


def test_audit_flags_ist_utc_conversion_mismatch(tmp_path):
    data = tmp_path / "pilot"
    make_fixture(data, bad_utc=True)
    result = audit(data)
    assert result["checks"]["ist_utc_conversion_consistent"] is False
    assert result["overall_pass"] is False


def test_audit_flags_observed_station_missing_from_registry(tmp_path):
    data = tmp_path / "pilot"
    make_fixture(data, unknown_station=True)
    result = audit(data)
    assert result["checks"]["all_observed_stations_have_registry_entry"] is False
    assert result["station_reconciliation"]["observed_ids_not_in_registry"] == ["site_missing"]


def test_audit_rejects_missing_parquet_partition(tmp_path):
    data = tmp_path / "pilot"
    data.mkdir()
    (data / "ingestion_manifest.json").write_text(json.dumps({
        "total_rows": 0, "partitions": [{"file": "missing.parquet"}]
    }))
    (data / "cpcb_station_registry.json").write_text('{"data":[]}')
    with pytest.raises(ValueError, match="missing"):
        audit(data)
