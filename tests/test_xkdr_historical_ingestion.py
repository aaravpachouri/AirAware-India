from datetime import date

import pytest

from scripts.ingest_xkdr_historical import month_range, parse_month, station_rows


def test_month_range_crosses_year_boundary():
    assert list(month_range((2024, 11), (2025, 2))) == [
        (2024, 11), (2024, 12), (2025, 1), (2025, 2)
    ]


def test_parse_month_rejects_invalid_values():
    assert parse_month("2024-02") == (2024, 2)
    with pytest.raises(Exception):
        parse_month("2024-13")
    with pytest.raises(Exception):
        parse_month("2024")


def test_station_rows_supports_provider_envelopes():
    assert station_rows({"stations": [{"station_id": "site_103"}]}) == [
        {"station_id": "site_103"}
    ]
    with pytest.raises(RuntimeError):
        station_rows({"unexpected": []})
