"""Summarise prospective same-station validation without overstating results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "prospective_station_validation.jsonl"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261110


def load_evaluated_records(path: Path = LEDGER) -> pd.DataFrame:
    """Load only completed forecast/observation pairs from the append-only ledger."""
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in prospective ledger line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Prospective ledger line {line_number} is not an object")
            if record.get("status") == "evaluated":
                rows.append(record)
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def _cluster_bootstrap_ci(frame: pd.DataFrame) -> dict[str, float] | None:
    """Paired percentile CI, resampling whole stations to respect repeated station rows."""
    station_key = "station.id"
    stations = frame[station_key].dropna().unique()
    if len(stations) < 5:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped = {station: frame.loc[frame[station_key] == station] for station in stations}
    improvements = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        sample = pd.concat([grouped[station] for station in sampled], ignore_index=True)
        improvements[index] = (
            sample["persistence_absolute_error"].mean()
            - sample["absolute_error"].mean()
        )
    low, high = np.percentile(improvements, [2.5, 97.5])
    return {"lower": round(float(low), 3), "upper": round(float(high), 3)}


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "evaluated_forecasts": 0,
            "stations": 0,
            "cities": 0,
            "model_mae": None,
            "persistence_mae": None,
            "mae_improvement": None,
            "station_cluster_bootstrap_95pct_ci": None,
            "interpretation": "No prospective same-station forecast has been evaluated yet.",
        }

    required = [
        "absolute_error", "persistence_absolute_error", "station.id",
        "station.city", "predicted_aqi", "observed_aqi", "actual_horizon_hours",
    ]
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError(f"Missing evaluated ledger fields: {missing}")

    valid = frame.copy()
    numeric = [
        "absolute_error", "persistence_absolute_error", "predicted_aqi",
        "observed_aqi", "actual_horizon_hours",
    ]
    for column in numeric:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=numeric + ["station.id"])
    if valid.empty:
        raise ValueError("Evaluated ledger rows contain no complete numeric forecast pairs")

    model_errors = valid["absolute_error"]
    baseline_errors = valid["persistence_absolute_error"]
    improvement = float(baseline_errors.mean() - model_errors.mean())
    wins = int((model_errors < baseline_errors).sum())
    losses = int((model_errors > baseline_errors).sum())
    ties = int((model_errors == baseline_errors).sum())

    by_station = []
    for station_id, group in valid.groupby("station.id", dropna=True):
        city_values = group["station.city"].dropna().astype(str)
        city = next((value for value in city_values if value.strip()), "Unspecified")
        station_name_values = group.get("station.name", pd.Series(dtype=str)).dropna().astype(str)
        by_station.append({
            "station_id": int(station_id),
            "station": station_name_values.iloc[0] if not station_name_values.empty else f"Station {station_id}",
            "city": city,
            "n": int(len(group)),
            "model_mae": round(float(group["absolute_error"].mean()), 3),
            "persistence_mae": round(float(group["persistence_absolute_error"].mean()), 3),
        })
    by_station.sort(key=lambda item: (item["city"], item["station"]))

    city_rows = []
    valid = valid.assign(_city=valid["station.city"].fillna("").astype(str).str.strip())
    valid.loc[valid["_city"] == "", "_city"] = "Unspecified"
    for city, group in valid.groupby("_city"):
        city_rows.append({
            "city": city,
            "stations": int(group["station.id"].nunique()),
            "n": int(len(group)),
            "model_mae": round(float(group["absolute_error"].mean()), 3),
            "persistence_mae": round(float(group["persistence_absolute_error"].mean()), 3),
        })
    city_rows.sort(key=lambda item: item["city"])

    ci = _cluster_bootstrap_ci(valid)
    if len(valid["station.id"].unique()) < 5:
        ci_note = "Not calculated: fewer than five distinct stations."
    else:
        ci_note = (
            "Exploratory paired percentile interval from 2,000 station-cluster bootstrap "
            "replicates; it is not evidence of generalisation by itself."
        )

    return {
        "evaluated_forecasts": int(len(valid)),
        "stations": int(valid["station.id"].nunique()),
        "cities": int(valid.loc[valid["_city"] != "Unspecified", "_city"].nunique()),
        "date_range_utc": {
            "first_issue": str(valid["issued_at"].min()) if "issued_at" in valid else None,
            "last_observation": str(valid["observed_at"].max()) if "observed_at" in valid else None,
        },
        "actual_horizon_hours": {
            "min": round(float(valid["actual_horizon_hours"].min()), 2),
            "median": round(float(valid["actual_horizon_hours"].median()), 2),
            "max": round(float(valid["actual_horizon_hours"].max()), 2),
        },
        "model_mae": round(float(model_errors.mean()), 3),
        "persistence_mae": round(float(baseline_errors.mean()), 3),
        "mae_improvement": round(improvement, 3),
        "model_rmse": round(float(np.sqrt(np.mean(np.square(model_errors)))), 3),
        "persistence_rmse": round(float(np.sqrt(np.mean(np.square(baseline_errors)))), 3),
        "model_bias_predicted_minus_observed": round(
            float((valid["predicted_aqi"] - valid["observed_aqi"]).mean()), 3
        ),
        "median_absolute_error": round(float(model_errors.median()), 3),
        "p90_absolute_error": round(float(model_errors.quantile(0.90)), 3),
        "paired_comparison": {"model_wins": wins, "model_losses": losses, "ties": ties},
        "station_cluster_bootstrap_95pct_ci_for_mae_improvement": ci,
        "bootstrap_note": ci_note,
        "per_station": by_station,
        "per_city": city_rows,
        "interpretation": (
            "Descriptive prospective results only. These are OpenAQ-derived station snapshots, "
            "not official CPCB 24-hour AQI. Repeated forecasts from the same station are "
            "clustered for the exploratory interval; report sample size and geographic coverage. "
            "A positive MAE improvement favours the model, but a small or geographically "
            "concentrated sample does not establish generalisation."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(load_evaluated_records()), indent=2, allow_nan=False))
