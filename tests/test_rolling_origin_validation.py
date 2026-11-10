"""Tests for leakage-safe expanding-window rolling-origin validation."""
import numpy as np
import pandas as pd
import pytest

from rolling_origin_validation import evaluate_rolling_origin, rolling_origin_splits


def sample_frame(days=100, cities=("A", "B", "C")):
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for city_index, city in enumerate(cities):
        for index, date in enumerate(dates):
            aqi = float(40 + city_index * 15 + index % 17)
            rows.append({
                "Date": date,
                "City": city,
                "PM2.5": 10.0 + index % 5,
                "PM10": 20.0 + index % 7,
                "NO2": 5.0,
                "CO": 0.5,
                "SO2": 2.0,
                "O3": 8.0,
                "AQI": aqi,
                "Previous_Day_AQI": aqi - 1,
                "AQI_3Day_Avg": aqi - 1,
                "AQI_Change": 1.0,
                "Day": date.day,
                "Month": date.month,
                "DayOfWeek": date.dayofweek,
                "Next_Day_AQI": aqi + 1,
            })
    return pd.DataFrame(rows)


def test_rolling_origin_folds_are_strictly_chronological_and_disjoint():
    frame = sample_frame()
    folds = list(rolling_origin_splits(frame, n_test_blocks=5))
    assert len(folds) == 5
    seen_test_dates = []
    previous_test_end = None
    for _, train, test in folds:
        assert train["Date"].max() < test["Date"].min() - pd.Timedelta(days=1)
        assert train["Date"].max() + pd.Timedelta(days=1) < test["Date"].min()
        if previous_test_end is not None:
            assert test["Date"].min() > previous_test_end
        previous_test_end = test["Date"].max()
        seen_test_dates.extend(test["Date"].unique())
    assert len(seen_test_dates) == len(set(seen_test_dates))


def test_rolling_origin_rejects_too_few_dates():
    with pytest.raises(ValueError, match="Insufficient dates"):
        list(rolling_origin_splits(sample_frame(days=8), n_test_blocks=5))


def test_evaluator_reports_baselines_and_candidate_metrics():
    frame = sample_frame()
    factories = {
        "constant-test-model": lambda: DummyRegressor(strategy="mean")
    }
    fold_metrics, city_metrics, summary = evaluate_rolling_origin(
        frame, n_test_blocks=4, model_factories=factories
    )
    assert summary["folds"] == 4
    assert summary["test_rows_total"] > 0
    assert {"Persistence", "Three-day mean", "constant-test-model"} <= set(fold_metrics["model"])
    assert set(city_metrics["city"]) == {"A", "B", "C"}
    assert "constant-test-model" in summary["mae_improvement_vs_persistence"]
