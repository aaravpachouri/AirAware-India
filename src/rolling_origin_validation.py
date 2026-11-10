"""Expanding-window rolling-origin evaluation for next-day AQI forecasting.

Every fold predicts a strictly later contiguous date block. Training labels end
before the first test feature date, so the target windows cannot overlap.
Imputation and model fitting happen inside each training fold only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evaluation_utils import FEATURES, load_ml_data
from model_benchmark import build_ridge_model
from train_model import build_model

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
INITIAL_TRAIN_FRACTION = 0.50
N_TEST_BLOCKS = 5
MODEL_FACTORIES = {
    "Random Forest (city-aware)": lambda: build_model(include_city=True),
    "Random Forest (city-agnostic)": lambda: build_model(include_city=False),
    "Ridge (city-agnostic)": lambda: build_ridge_model(include_city=False),
}


def rolling_origin_splits(
    frame: pd.DataFrame,
    initial_train_fraction: float = INITIAL_TRAIN_FRACTION,
    n_test_blocks: int = N_TEST_BLOCKS,
):
    """Yield expanding chronological folds with non-overlapping target windows."""
    if not 0.4 <= initial_train_fraction < 0.8:
        raise ValueError("initial_train_fraction must be in [0.4, 0.8).")
    if n_test_blocks < 2:
        raise ValueError("At least two rolling test blocks are required.")
    dates = pd.DatetimeIndex(
        pd.to_datetime(frame["Date"]).dropna().drop_duplicates().sort_values()
    )
    first_test = int(np.floor(len(dates) * initial_train_fraction))
    remaining = len(dates) - first_test
    if remaining < n_test_blocks * 2:
        raise ValueError("Insufficient dates for the requested rolling-origin folds.")
    blocks = np.array_split(dates[first_test:], n_test_blocks)
    for fold, block in enumerate(blocks, start=1):
        if len(block) == 0:
            continue
        test_start = pd.Timestamp(block[0])
        # Target for feature date D is AQI at D+1. Exclude D=test_start-1
        # from training to prevent its target from entering the test window.
        train = frame.loc[pd.to_datetime(frame["Date"]) < test_start - pd.Timedelta(days=1)].copy()
        test = frame.loc[pd.to_datetime(frame["Date"]).isin(block)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Fold {fold} produced an empty train or test set.")
        if train["Date"].max() + pd.Timedelta(days=1) >= test["Date"].min():
            raise AssertionError(f"Fold {fold} has overlapping target windows.")
        if train["Date"].max() >= test["Date"].min():
            raise AssertionError(f"Fold {fold} is not chronological.")
        yield fold, train, test


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
    }


def evaluate_rolling_origin(
    frame: pd.DataFrame,
    n_test_blocks: int = N_TEST_BLOCKS,
    model_factories: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return fold metrics, per-city metrics, and an auditable JSON summary."""
    factories = model_factories or MODEL_FACTORIES
    fold_rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    all_actual: list[float] = []
    all_predictions: dict[str, list[float]] = {
        "Persistence": [],
        "Three-day mean": [],
        **{name: [] for name in factories},
    }

    for fold, train, test in rolling_origin_splits(
        frame, n_test_blocks=n_test_blocks
    ):
        actual = test["Next_Day_AQI"].to_numpy(dtype=float)
        predictions: dict[str, np.ndarray] = {
            "Persistence": test["AQI"].to_numpy(dtype=float),
            "Three-day mean": test["AQI_3Day_Avg"].to_numpy(dtype=float),
        }
        for name, factory in factories.items():
            estimator = factory()
            estimator.fit(train[FEATURES], train["Next_Day_AQI"])
            predictions[name] = np.clip(estimator.predict(test[FEATURES]), 0, 500)

        all_actual.extend(actual.tolist())
        for name, values in predictions.items():
            all_predictions[name].extend(np.asarray(values, dtype=float).tolist())
            row = {
                "fold": fold,
                "model": name,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_start": str(pd.Timestamp(train["Date"].min()).date()),
                "train_end": str(pd.Timestamp(train["Date"].max()).date()),
                "test_start": str(pd.Timestamp(test["Date"].min()).date()),
                "test_end": str(pd.Timestamp(test["Date"].max()).date()),
                **_metrics(actual, np.asarray(values, dtype=float)),
            }
            fold_rows.append(row)
            scored = test[["City"]].copy()
            scored["actual"] = actual
            scored["predicted"] = values
            for city, group in scored.groupby("City", sort=True):
                city_rows.append({
                    "fold": fold,
                    "model": name,
                    "city": str(city),
                    "n": int(len(group)),
                    "mae": round(float(mean_absolute_error(group["actual"], group["predicted"])), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(group["actual"], group["predicted"]))), 4),
                })

    actual_all = np.asarray(all_actual, dtype=float)
    overall = {
        name: _metrics(actual_all, np.asarray(values, dtype=float))
        for name, values in all_predictions.items()
    }
    baseline_mae = overall["Persistence"]["mae"]
    improvements = {
        name: round(baseline_mae - metrics["mae"], 4)
        for name, metrics in overall.items()
        if name != "Persistence"
    }
    summary = {
        "protocol": "Expanding-window rolling-origin; contiguous date blocks; train-only preprocessing; one-day target gap.",
        "folds": int(len(set(row["fold"] for row in fold_rows))),
        "test_rows_total": int(len(actual_all)),
        "overall_metrics": overall,
        "mae_improvement_vs_persistence": improvements,
        "interpretation": (
            "Positive improvement favours the candidate model. These folds estimate "
            "performance across later dates for cities represented in the historical "
            "dataset; they do not establish transfer to unseen cities or station-level "
            "live accuracy."
        ),
    }
    return pd.DataFrame(fold_rows), pd.DataFrame(city_rows), summary


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = load_ml_data()
    folds, cities, summary = evaluate_rolling_origin(frame)
    folds.to_csv(REPORTS / "rolling_origin_fold_metrics.csv", index=False)
    cities.to_csv(REPORTS / "rolling_origin_city_metrics.csv", index=False)
    (REPORTS / "rolling_origin_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
