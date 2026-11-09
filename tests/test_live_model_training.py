"""Regression tests for the live forecast training pipeline."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_live_model import LIVE_FEATURES, build_live_model


def test_live_model_pipeline_handles_missing_pollutants_without_zero_filling():
    frame = pd.DataFrame({
        "AQI": np.linspace(20, 200, 30),
        "PM2.5": np.linspace(5, 100, 30),
        "PM10": np.linspace(10, 180, 30),
        "Day": [((i % 28) + 1) for i in range(30)],
        "Month": [((i % 12) + 1) for i in range(30)],
        "DayOfWeek": [i % 7 for i in range(30)],
    })
    frame.loc[[2, 8], "PM2.5"] = np.nan
    frame.loc[[4, 11], "PM10"] = np.nan
    target = frame["AQI"] * 0.8 + 10

    model = build_live_model()
    model.fit(frame[LIVE_FEATURES], target)
    predictions = model.predict(frame[LIVE_FEATURES])

    assert len(predictions) == len(frame)
    assert np.isfinite(predictions).all()
