def test_remaining_month_returns_all_dates_and_marks_unvalidated_horizons(tmp_path, monkeypatch):
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    rows = history(42, now-timedelta(hours=240), 241)
    path = tmp_path/"history.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    report = {
        "schema_version": 1,
        "validated_horizons_days": [1],
        "horizons": {"1": {"forecast_enabled": True, "status": "validated"}},
    }
    (tmp_path/"report.json").write_text(json.dumps(report), encoding="utf-8")

    class Model:
        def predict(self, frame):
            assert len(frame) == 1
            return [123.0]

    monkeypatch.setattr(monthly.joblib, "load", lambda _path: {
        "validation_status": "validated",
        "models": {1: {"model": Model(), "validation_status": "validated",
                       "residual_p90": 12}},
    })
    def varying_daily_aqi(series, anchor):
        day_offset = (anchor.date() - now.date()).days
        return (max(0.0, min(500.0, 100.0 + 10.0 * day_offset)), {})

    monkeypatch.setattr(monthly, "aqi_for_window", varying_daily_aqi)
    result = monthly.predict_remaining_month(
        42, issued_at=now, timezone_name="Asia/Kolkata",
        history_path=path, model_path=tmp_path/"model.joblib",
        report_path=tmp_path/"report.json",
    )
    assert result["available"] is True
    assert result["days"][0]["date"] == "2026-11-07"
    assert result["days"][1]["date"] == "2026-11-08"
    assert result["days"][1]["aqi"] == 123.0
    assert result["days"][1]["lower_90"] == 111.0
    assert result["days"][-1]["date"] == "2026-12-07"
    assert result["days"][2]["status"] == "trend_reference"
    assert result["days"][2]["aqi"] is not None
    assert result["days"][2]["baseline"] == "recent_daily_trend"
    assert all(d["status"] == "trend_reference" for d in result["days"][2:])
    assert len({d["aqi"] for d in result["days"][2:]}) > 1


def test_trend_reference_changes_with_recent_daily_direction():
    history_values = [(day, value) for day, value in enumerate([50, 65, 80, 95, 110, 125, 140])]
    forecasts = [monthly._trend_reference(140, history_values, horizon)
                 for horizon in (1, 2, 3, 7, 14, 30)]
    assert len(set(forecasts)) == len(forecasts)
    assert forecasts[1] > forecasts[0]
    assert forecasts[30] < forecasts[14]


def test_forecast_ledger_is_append_only_and_scores_only_matured_targets(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = history(7, start, 24*18)
    class Model:
        def predict(self, frame):
            return [80.0]
    artifact = {"trained_at": "v1", "models": {
        "1": {"model": Model(), "residual_p90": 10}
    }}
    ledger = tmp_path/"ledger.jsonl"
    issued = datetime(2026, 1, 5, tzinfo=timezone.utc)
    first = monthly.update_forecast_ledger(rows, artifact, ledger_path=ledger, now=issued)
    again = monthly.update_forecast_ledger(rows, artifact, ledger_path=ledger, now=issued)
    assert first["forecast_events_added"] == 1
    assert again["forecast_events_added"] == 0
    matured = monthly.update_forecast_ledger(
        rows, artifact, ledger_path=ledger, now=issued+timedelta(days=2)
    )
    assert matured["verification_events_added"] == 1
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    forecast = next(event for event in events if event["event_type"] == "forecast")
    verification = next(event for event in events if event["event_type"] == "verification")
    assert forecast["horizon_days"] == 1
    assert forecast["input_data_cutoff"] == issued.isoformat()
    assert verification["forecast_event_id"] == forecast["event_id"]
    assert verification["absolute_error"] >= 0



def test_month_calendar_is_rendered_even_before_any_horizon_is_validated(tmp_path):
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    rows = history(42, now-timedelta(hours=240), 241)
    path = tmp_path/"history.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    result = monthly.predict_remaining_month(
        42, issued_at=now, timezone_name="Asia/Kolkata",
        history_path=path, model_path=tmp_path/"missing.joblib",
        report_path=tmp_path/"missing-report.json",
    )
    assert result["available"] is True
    assert len(result["days"]) == 31  # today plus 30 forecast dates
    assert result["days"][0]["date"] == "2026-11-07"
    assert result["days"][-1]["date"] == "2026-12-07"
    assert all(day["aqi"] is not None for day in result["days"][1:])
    assert all(day["status"] == "baseline_forecast" for day in result["days"][1:])



def test_monthly_report_serializes_numpy_station_ids(tmp_path, monkeypatch):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    records = []
    for station_id in range(1, 51):
        records.extend([
            {"station_id": station_id, "timestamp": (start - pd.Timedelta(days=1)).isoformat()},
            {"station_id": station_id, "timestamp": (start + pd.Timedelta(days=40)).isoformat()},
        ])
        for day in range(10):
            for hour in range(24):
                issued = start + pd.Timedelta(days=day, hours=hour)
                rows.append({
                    "station_id": station_id,
                    "issued_at": issued.isoformat(),
                    **{feature: 50.0 for feature in FEATURES},
                    "target_aqi": 55.0,
                })
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(monthly, "build_horizon_frame", lambda _records, _horizon, **_kwargs: frame.copy())

    class DummyModel:
        def fit(self, _x, _y):
            return self

        def predict(self, x):
            return [50.0] * len(x)

    monkeypatch.setattr(monthly, "_pipeline", DummyModel)
    monkeypatch.setattr(monthly, "_metrics", lambda _frame, _prediction: {
        "samples": len(_frame), "stations": int(_frame.station_id.nunique()),
        "model_mae": 1.0, "persistence_mae": 10.0,
        "three_day_mean_mae": 9.0, "best_baseline_mae": 9.0,
        "improvement_over_best_baseline_pct": 88.889,
        "absolute_error_p90": 1.0,
    })

    # This regression targets JSON report serialization, not sklearn/joblib pickling.
    # The local dummy estimator is intentionally not pickleable.
    monkeypatch.setattr(monthly.joblib, "dump", lambda *_args, **_kwargs: None)

    report = monthly.train_monthly_models(
        records,
        model_path=tmp_path / "monthly.joblib",
        report_path=tmp_path / "report.json",
        max_horizon_days=1,
    )

    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["horizons"]["1"]["status"] == "validated"
    assert all(type(value) is int for value in saved["horizons"]["1"]["station_holdout_ids"])


def test_rolling_outlook_uses_fresh_snapshot_as_reference_when_hourly_coverage_fails(tmp_path):
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    result = monthly.predict_remaining_month(
        77, issued_at=now, timezone_name="Asia/Kolkata",
        history_path=tmp_path/"missing-history.jsonl",
        model_path=tmp_path/"missing-model.joblib",
        report_path=tmp_path/"missing-report.json",
        current_snapshot_aqi=140, snapshot_age_hours=1.25,
    )
    assert result["snapshot_reference_ready"] is True
    assert result["data_age_hours"] == pytest.approx(1.25)
    assert result["days"][0]["aqi"] == 140
    assert result["days"][0]["status"] == "snapshot_estimate"
    assert result["days"][1]["aqi"] == 140
    assert result["days"][1]["status"] == "snapshot_reference"
    assert result["days"][1]["lower_90"] is None
    assert result["days"][-1]["horizon_days"] == 30


def test_stale_snapshot_is_never_used_as_monthly_reference(tmp_path):
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    result = monthly.predict_remaining_month(
        77, issued_at=now, history_path=tmp_path/"missing-history.jsonl",
        model_path=tmp_path/"missing-model.joblib",
        report_path=tmp_path/"missing-report.json",
        current_snapshot_aqi=140, snapshot_age_hours=15,
    )
    assert result["snapshot_reference_ready"] is False
    assert all(day["aqi"] is None for day in result["days"])
