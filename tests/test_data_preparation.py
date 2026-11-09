"""Tests for calendar-aware AQI feature engineering."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_ml_data


def source_frame(dates):
    rows = []
    for index, date in enumerate(pd.to_datetime(dates)):
        rows.append({
            "City": "Test City",
            "Date": date,
            "PM2.5": 10.0 + index,
            "PM10": 20.0 + index,
            "NO2": 5.0,
            "CO": 0.5,
            "SO2": 2.0,
            "O3": 12.0,
            "AQI": 40.0 + index,
        })
    return pd.DataFrame(rows)


def test_three_day_average_does_not_bridge_calendar_gaps(tmp_path, monkeypatch):
    dates = [
        "2020-01-01",
        "2020-01-02",
        "2020-01-04",
        "2020-01-05",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
    ]
    input_path = tmp_path / "cleaned.csv"
    output_path = tmp_path / "ml.csv"
    source_frame(dates).to_csv(input_path, index=False)

    monkeypatch.setattr(prepare_ml_data, "INPUT_PATH", input_path)
    monkeypatch.setattr(prepare_ml_data, "OUTPUT_PATH", output_path)
    prepare_ml_data.main()

    result = pd.read_csv(output_path, parse_dates=["Date"])
    result_dates = set(result["Date"].dt.strftime("%Y-%m-%d"))

    # Jan 4 and Jan 5 have a three-row window spanning a missing calendar day.
    assert "2020-01-04" not in result_dates
    assert "2020-01-05" not in result_dates

    # Jan 6 and Jan 7 have three consecutive observations and a next-day target.
    assert {"2020-01-06", "2020-01-07"}.issubset(result_dates)


def test_duplicate_city_date_rows_fail_closed(tmp_path, monkeypatch):
    frame = source_frame(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    input_path = tmp_path / "cleaned.csv"
    output_path = tmp_path / "ml.csv"
    frame.to_csv(input_path, index=False)

    monkeypatch.setattr(prepare_ml_data, "INPUT_PATH", input_path)
    monkeypatch.setattr(prepare_ml_data, "OUTPUT_PATH", output_path)

    with pytest.raises(ValueError, match="Duplicate city/date observations"):
        prepare_ml_data.main()
