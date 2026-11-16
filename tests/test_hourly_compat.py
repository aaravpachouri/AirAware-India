from __future__ import annotations

import pytest

from hourly_compat import fetch_hourly_station_data


def test_requests_extended_station_history_when_supported():
    calls = []

    def fetcher(location_id, *, history_days=1):
        calls.append((location_id, history_days))
        return {"eligible": True, "hourly_observations": [1]}

    result = fetch_hourly_station_data(fetcher, 123, history_days=30)

    assert calls == [(123, 30)]
    assert result["eligible"] is True
    assert result["hourly_observations"] == [1]


def test_retries_legacy_signature_without_history_days():
    calls = []

    def legacy_fetcher(location_id):
        calls.append(location_id)
        return {"eligible": True, "hourly_observations": [1]}

    result = fetch_hourly_station_data(legacy_fetcher, 456, history_days=30)

    assert calls == [456]
    assert result["eligible"] is True


def test_does_not_hide_unrelated_type_errors():
    def broken_fetcher(location_id, *, history_days=1):
        raise TypeError("internal conversion failed")

    with pytest.raises(TypeError, match="internal conversion failed"):
        fetch_hourly_station_data(broken_fetcher, 789, history_days=30)


def test_does_not_hide_provider_errors():
    def broken_fetcher(location_id, *, history_days=1):
        raise RuntimeError("OpenAQ timeout")

    with pytest.raises(RuntimeError, match="OpenAQ timeout"):
        fetch_hourly_station_data(broken_fetcher, 789, history_days=30)
