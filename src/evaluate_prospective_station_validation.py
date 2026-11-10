"""Summarise prospective same-station validation without overstating results."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "prospective_station_validation.jsonl"

def load_evaluated_records(path: Path = LEDGER) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("status") == "evaluated":
                    rows.append(record)
    return pd.json_normalize(rows) if rows else pd.DataFrame()

def summarize(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"evaluated_forecasts": 0, "stations": 0, "cities": 0,
                "model_mae": None, "persistence_mae": None, "mae_improvement": None,
                "interpretation": "No prospective same-station forecast has been evaluated yet."}
    required = ["absolute_error", "persistence_absolute_error", "station.id", "station.city"]
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError(f"Missing evaluated ledger fields: {missing}")
    valid = frame.dropna(subset=["absolute_error", "persistence_absolute_error"])
    model_mae = float(valid["absolute_error"].mean())
    baseline_mae = float(valid["persistence_absolute_error"].mean())
    return {
        "evaluated_forecasts": int(len(valid)),
        "stations": int(valid["station.id"].nunique()),
        "cities": int(valid["station.city"].replace("", pd.NA).nunique()),
        "model_mae": round(model_mae, 3),
        "persistence_mae": round(baseline_mae, 3),
        "mae_improvement": round(baseline_mae - model_mae, 3),
        "interpretation": "Descriptive prospective results only. These are OpenAQ-derived station snapshots, not official CPCB 24-hour AQI. Small samples do not establish generalisation.",
    }

if __name__ == "__main__":
    print(json.dumps(summarize(load_evaluated_records()), indent=2))
