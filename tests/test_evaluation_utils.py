"""Regression tests for AirAware's shared temporal evaluation split."""
import pandas as pd
import pytest

from src.evaluation_utils import FEATURES, chronological_split


def make_frame(days: int = 20) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2020-01-01", periods=days, freq="D"):
        row = {"Date": day, "City": "Test City", "Next_Day_AQI": 80.0}
        row.update({feature: 1.0 for feature in FEATURES if feature != "City"})
        rows.append(row)
    return pd.DataFrame(rows)


def test_split_keeps_train_and_test_chronological_and_target_windows_disjoint():
    train, test, test_start = chronological_split(make_frame())

    assert not train.empty
    assert not test.empty
    assert train["Date"].max() < test_start
    assert test["Date"].min() == test_start
    # The target for the last training feature date is the next calendar day.
    assert train["Date"].max() + pd.Timedelta(days=1) < test["Date"].min()


def test_split_is_deterministic():
    frame = make_frame()
    train_a, test_a, start_a = chronological_split(frame)
    train_b, test_b, start_b = chronological_split(frame)

    assert start_a == start_b
    assert train_a["Date"].tolist() == train_b["Date"].tolist()
    assert test_a["Date"].tolist() == test_b["Date"].tolist()


def test_split_rejects_too_few_distinct_dates():
    with pytest.raises(ValueError, match="four distinct dates"):
        chronological_split(make_frame(days=3))


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_split(make_frame(), train_fraction=1.0)
