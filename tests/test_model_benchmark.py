"""Tests for the model-family benchmark pipelines."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation_utils import FEATURES
from model_benchmark import build_ridge_model


def synthetic_frame():
    rows = []
    for i in range(40):
        row = {"City": f"City {i % 4}", "Next_Day_AQI": float(30 + i * 2)}
        for j, feature in enumerate(FEATURES):
            if feature != "City":
                row[feature] = float((i + 1) * (j + 1) % 97)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[[3, 11], "PM2.5"] = np.nan
    frame.loc[[7, 19], "NO2"] = np.nan
    return frame


def test_ridge_pipelines_fit_with_and_without_city_feature():
    frame = synthetic_frame()
    for include_city in (True, False):
        model = build_ridge_model(include_city=include_city)
        model.fit(frame[FEATURES], frame["Next_Day_AQI"])
        predictions = model.predict(frame[FEATURES])
        assert len(predictions) == len(frame)
        assert np.isfinite(predictions).all()
