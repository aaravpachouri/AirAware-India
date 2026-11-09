"""Tests for city-group generalization splits."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation_utils import city_holdout_split


def multi_city_frame(city_count=5):
    rows = []
    for city_index in range(city_count):
        for date in pd.date_range("2020-01-01", periods=10, freq="D"):
            rows.append({
                "Date": date,
                "City": f"City {city_index}",
                "Next_Day_AQI": 50.0 + city_index,
            })
    return pd.DataFrame(rows)


def test_city_holdout_has_no_city_overlap():
    train, test, held_out = city_holdout_split(multi_city_frame())

    assert set(train["City"]).isdisjoint(set(test["City"]))
    assert set(test["City"]) == set(held_out)
    assert not train.empty
    assert not test.empty


def test_city_holdout_is_deterministic():
    frame = multi_city_frame()
    train_a, test_a, held_a = city_holdout_split(frame)
    train_b, test_b, held_b = city_holdout_split(frame)

    assert held_a == held_b
    assert train_a["City"].unique().tolist() == train_b["City"].unique().tolist()
    assert test_a["City"].unique().tolist() == test_b["City"].unique().tolist()


def test_city_holdout_requires_multiple_cities():
    with pytest.raises(ValueError, match="At least two distinct cities"):
        city_holdout_split(multi_city_frame(city_count=1))
