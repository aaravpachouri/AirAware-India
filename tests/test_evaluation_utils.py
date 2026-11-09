"""Regression tests for AirAware's temporal split and missing-value handling."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluation_utils
from evaluation_utils import FEATURES, chronological_split
from train_model import build_model


EXPECTED_DASHBOARD_FEATURE_ORDER = [
    "City",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI",
    "Previous_Day_AQI",
    "AQI_3Day_Avg",
    "AQI_Change",
    "Day",
    "Month",
    "DayOfWeek",
]


def make_frame(days: int = 20) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2020-01-01", periods=days, freq="D"):
        row = {"Date": day, "City": "Test City", "Next_Day_AQI": 80.0}
        row.update({feature: 1.0 for feature in FEATURES if feature != "City"})
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_order_matches_historical_dashboard_contract():
    assert FEATURES == EXPECTED_DASHBOARD_FEATURE_ORDER


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


def test_loader_preserves_missing_pollutant_values_for_train_only_imputation(tmp_path, monkeypatch):
    frame = make_frame()
    frame.loc[3, "PM2.5"] = np.nan
    data_path = tmp_path / "ml_data.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setattr(evaluation_utils, "DATA_PATH", data_path)

    loaded = evaluation_utils.load_ml_data()

    assert loaded["PM2.5"].isna().sum() == 1
    assert len(loaded) == len(frame)


def test_model_fits_imputer_on_training_rows_and_predicts_with_missing_pollutants():
    frame = make_frame()
    frame["PM2.5"] = np.arange(len(frame), dtype=float)
    frame.loc[[2, 5, 18], "PM2.5"] = np.nan
    train, test, _ = chronological_split(frame)

    model = build_model()
    model.fit(train[FEATURES], train["Next_Day_AQI"])
    predictions = model.predict(test[FEATURES])

    imputer = model.named_steps["preprocessor"].named_transformers_["pollutant_imputer"]
    expected_training_median = train["PM2.5"].median()
    assert imputer.statistics_[0] == pytest.approx(expected_training_median)
    assert len(predictions) == len(test)
    assert np.isfinite(predictions).all()
