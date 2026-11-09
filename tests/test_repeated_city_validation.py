"""Tests for repeated city-group validation and paired bootstrap intervals."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repeated_city_validation import bootstrap_mean_interval, iter_city_folds


def make_city_frame(city_count=10, rows_per_city=4):
    return pd.DataFrame(
        [
            {"City": f"City {city}", "value": row}
            for city in range(city_count)
            for row in range(rows_per_city)
        ]
    )


def test_each_city_is_held_out_exactly_once_without_overlap():
    frame = make_city_frame()
    seen_test_cities = []
    for train_indices, test_indices in iter_city_folds(frame, n_splits=5):
        train_cities = set(frame.iloc[train_indices]["City"])
        test_cities = set(frame.iloc[test_indices]["City"])
        assert train_cities.isdisjoint(test_cities)
        seen_test_cities.extend(test_cities)
    assert sorted(seen_test_cities) == sorted(frame["City"].unique())


def test_city_fold_count_must_not_exceed_number_of_cities():
    with pytest.raises(ValueError, match="number of cities"):
        list(iter_city_folds(make_city_frame(city_count=3), n_splits=5))


def test_bootstrap_interval_is_deterministic_and_finite():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = bootstrap_mean_interval(values, n_bootstrap=1000, random_state=7)
    second = bootstrap_mean_interval(values, n_bootstrap=1000, random_state=7)
    assert first == second
    assert np.isfinite(first).all()
    assert first[0] <= np.mean(values) <= first[1]


def test_bootstrap_interval_rejects_empty_values():
    with pytest.raises(ValueError, match="finite city-level"):
        bootstrap_mean_interval([np.nan, np.inf])
