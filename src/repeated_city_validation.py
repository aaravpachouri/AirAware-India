"""Repeated city-group validation with paired city-level bootstrap intervals.

Each city appears in exactly one test fold. All candidate models are evaluated
against persistence and the three-day mean on identical rows. Bootstrap
intervals resample cities (not individual correlated city-day rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from evaluation_utils import FEATURES, load_ml_data
from model_benchmark import build_ridge_model
from train_model import build_model


def iter_city_folds(df: pd.DataFrame, n_splits: int = 5):
    """Yield deterministic group folds with no city overlap."""
    groups = df["City"].astype(str)
    city_count = groups.nunique()
    if n_splits < 2 or n_splits > city_count:
        raise ValueError(
            f"n_splits must be between 2 and the number of cities ({city_count})."
        )
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(df, groups=groups)


def bootstrap_mean_interval(
    values,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile interval for a paired city-level mean difference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("At least one finite city-level value is required.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    return float(lower), float(upper)


def model_factories():
    """Return fresh estimators for each fold."""
    return {
        "Random Forest (city-aware)": lambda: build_model(include_city=True),
        "Random Forest (city-agnostic)": lambda: build_model(include_city=False),
        "Ridge (city-aware)": lambda: build_ridge_model(include_city=True),
        "Ridge (city-agnostic)": lambda: build_ridge_model(include_city=False),
    }


def evaluate_repeated_city_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Evaluate candidates on repeated city groups and return macro-city metrics."""
    city_records: dict[str, dict[str, list[float]]] = {}
    factories = model_factories()

    for fold_number, (train_indices, test_indices) in enumerate(
        iter_city_folds(df, n_splits=n_splits), start=1
    ):
        train = df.iloc[train_indices]
        test = df.iloc[test_indices]
        train_cities = set(train["City"].astype(str))
        test_cities = set(test["City"].astype(str))
        if train_cities.intersection(test_cities):
            raise AssertionError("City leakage detected between train and test.")

        print(
            f"Fold {fold_number}/{n_splits}: "
            f"train={len(train)} rows, test={len(test)} rows, "
            f"held-out cities={', '.join(sorted(test_cities))}"
        )

        actual = test["Next_Day_AQI"].to_numpy(dtype=float)
        baselines = {
            "Persistence": test["AQI"].to_numpy(dtype=float),
            "Three-day mean": test["AQI_3Day_Avg"].to_numpy(dtype=float),
        }
        predictions = {}
        for name, factory in factories.items():
            estimator = factory()
            estimator.fit(train[FEATURES], train["Next_Day_AQI"])
            predictions[name] = np.clip(
                estimator.predict(test[FEATURES]), 0, 500
            )

        scored = test[["City"]].copy()
        scored["actual"] = actual
        for name, values in {**baselines, **predictions}.items():
            scored[name] = values

        for city, group in scored.groupby("City", sort=True):
            record = city_records.setdefault(
                str(city),
                {name: [] for name in ["actual", *baselines, *predictions]},
            )
            for name in record:
                record[name].extend(group[name].to_numpy(dtype=float).tolist())

    rows = []
    for model_name in [*factories, *baselines]:
        city_maes = []
        paired_improvements = []
        for city, record in sorted(city_records.items()):
            actual = np.asarray(record["actual"], dtype=float)
            predicted = np.asarray(record[model_name], dtype=float)
            city_mae = mean_absolute_error(actual, predicted)
            city_maes.append(city_mae)
            if model_name in factories:
                persistence_mae = mean_absolute_error(
                    actual, np.asarray(record["Persistence"], dtype=float)
                )
                paired_improvements.append(persistence_mae - city_mae)

        row = {
            "model": model_name,
            "cities": len(city_maes),
            "macro_city_mae": float(np.mean(city_maes)),
            "median_city_mae": float(np.median(city_maes)),
        }
        if model_name in factories:
            lower, upper = bootstrap_mean_interval(
                paired_improvements, n_bootstrap=n_bootstrap
            )
            row["mean_improvement_vs_persistence"] = float(
                np.mean(paired_improvements)
            )
            row["improvement_ci_95_lower"] = lower
            row["improvement_ci_95_upper"] = upper
        else:
            row["mean_improvement_vs_persistence"] = np.nan
            row["improvement_ci_95_lower"] = np.nan
            row["improvement_ci_95_upper"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    df = load_ml_data()
    print("REPEATED CITY-GROUP VALIDATION")
    print(f"Rows: {len(df)} | Cities: {df['City'].nunique()} | Folds: 5")
    results = evaluate_repeated_city_folds(df, n_splits=5)
    print("\nMacro-average across held-out cities (lower MAE is better):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
