import pytest

from live_air import calculate_subindex


@pytest.mark.parametrize(
    ("pollutant", "concentration", "expected"),
    [
        ("pm25", 30.5, 50.8333333333),
        ("pm10", 50.5, 50.5),
        ("no2", 40.5, 50.625),
        ("so2", 40.5, 50.625),
        ("o3", 50.5, 50.5),
        ("co", 1.05, 52.5),
    ],
)
def test_fractional_concentrations_are_interpolated_without_breakpoint_gaps(
    pollutant, concentration, expected
):
    assert calculate_subindex(pollutant, concentration) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pollutant", "concentration", "expected"),
    [
        ("pm25", 250, 400),
        ("pm10", 430, 400),
        ("no2", 400, 400),
        ("so2", 1600, 400),
        ("o3", 748, 400),
        ("co", 34, 400),
    ],
)
def test_top_cpcb_band_starts_at_the_published_severe_threshold(
    pollutant, concentration, expected
):
    assert calculate_subindex(pollutant, concentration) == pytest.approx(expected)
