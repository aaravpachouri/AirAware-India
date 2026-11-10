from __future__ import annotations

from typing import Any

EXPECTED_AVERAGING_PERIODS = {
    'pm25': '24h', 'pm10': '24h', 'no2': '24h', 'so2': '24h',
    'co': '8h', 'o3': '8h',
}


def assess_cpcb_aqi_eligibility(
    pollutant_averages: dict[str, float | int | None],
    coverage: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    '''Require prescribed time averages and verified coverage before AQI use.

    Latest-value snapshots are not accepted as AQI inputs. Coverage metadata
    must be produced by a time-series aggregation step, not inferred here.
    '''
    coverage = coverage or {}
    eligible_pollutants: list[str] = []
    rejected: dict[str, str] = {}

    for pollutant, expected_period in EXPECTED_AVERAGING_PERIODS.items():
        value = pollutant_averages.get(pollutant)
        if value is None:
            continue
        metadata = coverage.get(pollutant) or {}
        if metadata.get('averaging_period') != expected_period:
            rejected[pollutant] = f'requires {expected_period} average'
            continue
        if metadata.get('coverage_sufficient') is not True:
            rejected[pollutant] = 'observation coverage not verified'
            continue
        eligible_pollutants.append(pollutant)

    has_particulate = any(p in eligible_pollutants for p in ('pm25', 'pm10'))
    eligible = len(eligible_pollutants) >= 3 and has_particulate
    reasons = []
    if len(eligible_pollutants) < 3:
        reasons.append('at least three pollutants with validated averaging are required')
    if not has_particulate:
        reasons.append('PM2.5 or PM10 must be among the validated pollutants')

    return {
        'eligible': eligible,
        'eligible_pollutants': eligible_pollutants,
        'rejected_pollutants': rejected,
        'reasons': reasons,
        'status': 'eligible' if eligible else 'insufficient_validated_coverage',
    }
