"""Compatibility wrapper for live hourly fetchers during rolling deployments."""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any


_HISTORY_DAYS_KEYWORD_ERROR = re.compile(
    r"unexpected keyword argument\s+['\"]history_days['\"]"
)


def fetch_hourly_station_data(
    fetcher: Callable[..., dict[str, Any]],
    location_id: int,
    *,
    history_days: int = 30,
) -> dict[str, Any]:
    """Request longer history, falling back only for the known old signature.

    Do not swallow provider/API errors or unrelated TypeErrors. The legacy
    call still returns its normal rolling-24-hour aggregation; it simply
    cannot supply the optional extended history used by forecasting.
    """
    try:
        return fetcher(location_id, history_days=history_days)
    except TypeError as exc:
        if not _HISTORY_DAYS_KEYWORD_ERROR.search(str(exc)):
            raise
        return fetcher(location_id)
