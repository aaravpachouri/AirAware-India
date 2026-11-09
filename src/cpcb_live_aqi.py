"""CPCB-style live AQI aggregation from timestamped hourly observations.

This module is intentionally pure: the API adapter supplies hourly observations
in canonical CPCB breakpoint units, and this module validates time coverage.
It does not claim that OpenAQ data is official CPCB data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

AVERAGING_PERIOD_HOURS = {
    "pm25": 24,
    "pm10": 24,
    "no2": 24,
    "so2": 24,
    "co": 8,
    "o3": 8,
}
MIN_VALID_HOURS_IN_24H = 16
MIN_CONSECUTIVE_HOURS_FOR_8H = 8


def _utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def aggregate_hourly_observations(
    observations: dict[str, list[dict[str, Any]]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate recent hourly observations with explicit coverage checks.

    Each observation is {"timestamp": aware/ISO datetime, "value": float}.
    24-hour pollutants use the mean of valid hourly values in the rolling
    24-hour window and require at least 16 valid hours. CO/O3 use the latest
    complete 8-consecutive-hour mean and also require at least 16 valid hours
    in the 24-hour window. Duplicate hour buckets keep the latest valid value.
    """
    end = _utc_datetime(now or datetime.now(timezone.utc))
    start = end - timedelta(hours=24)
    output: dict[str, Any] = {
        "averages": {},
        "coverage": {},
        "eligible_pollutants": [],
        "rejected_pollutants": {},
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
    }

    for pollutant, period in AVERAGING_PERIOD_HOURS.items():
        by_hour: dict[datetime, float] = {}
        for item in observations.get(pollutant, []):
            try:
                stamp = _utc_datetime(item["timestamp"])
                value = float(item["value"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if stamp < start or stamp > end or value < 0:
                continue
            # Use the hour-ending timestamp as the bucket; duplicate readings
            # in one bucket are not counted as multiple coverage hours.
            hour = stamp.replace(minute=0, second=0, microsecond=0)
            if value != float("inf") and value == value:
                by_hour[hour] = value

        count = len(by_hour)
        coverage: dict[str, Any] = {
            "averaging_period": f"{period}h",
            "valid_hours_in_24h": count,
            "required_hours_in_24h": MIN_VALID_HOURS_IN_24H,
            "coverage_sufficient": count >= MIN_VALID_HOURS_IN_24H,
        }
        if count < MIN_VALID_HOURS_IN_24H:
            output["coverage"][pollutant] = coverage
            output["rejected_pollutants"][pollutant] = (
                f"only {count} valid hourly observations in 24h; "
                f"{MIN_VALID_HOURS_IN_24H} required"
            )
            continue

        ordered = sorted(by_hour.items())
        if period == 24:
            value = mean(v for _, v in ordered)
        else:
            # Find the most recent window of eight consecutive hour buckets.
            value = None
            for idx in range(len(ordered) - MIN_CONSECUTIVE_HOURS_FOR_8H, -1, -1):
                candidate = ordered[idx:idx + MIN_CONSECUTIVE_HOURS_FOR_8H]
                if len(candidate) != MIN_CONSECUTIVE_HOURS_FOR_8H:
                    continue
                if all(
                    candidate[j][0] - candidate[j - 1][0] == timedelta(hours=1)
                    for j in range(1, len(candidate))
                ):
                    value = mean(v for _, v in candidate)
                    break
            if value is None:
                coverage["coverage_sufficient"] = False
                output["coverage"][pollutant] = coverage
                output["rejected_pollutants"][pollutant] = (
                    "no complete 8-consecutive-hour window available"
                )
                continue

        output["averages"][pollutant] = value
        output["coverage"][pollutant] = coverage
        output["eligible_pollutants"].append(pollutant)

    has_particulate = any(
        p in output["eligible_pollutants"] for p in ("pm25", "pm10")
    )
    output["eligible"] = len(output["eligible_pollutants"]) >= 3 and has_particulate
    output["status"] = (
        "eligible" if output["eligible"]
        else "insufficient_validated_coverage"
    )
    output["reasons"] = []
    if len(output["eligible_pollutants"]) < 3:
        output["reasons"].append(
            "at least three pollutants need validated averaging and coverage"
        )
    if not has_particulate:
        output["reasons"].append(
            "PM2.5 or PM10 must have validated averaging and coverage"
        )
    return output
