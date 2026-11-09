import joblib
import pandas as pd

from live_air import (
    get_live_air_quality,
    estimate_cpcb_style_aqi,
    get_aqi_category
)


MODEL_PATH = "models/airaware_live_rf.joblib"


def predict_tomorrow():

    data = get_live_air_quality()

    pollutants = data["pollutants"]

    aqi_result = estimate_cpcb_style_aqi(
    pollutants
)

    current_aqi = aqi_result["aqi"]
    subindices = aqi_result["subindices"]

    if current_aqi is None:
        raise RuntimeError(
            "Could not calculate estimated AQI."
        )

    timestamp = pd.Timestamp(
        data["timestamp"]
    )

    from live_forecast_features import build_live_forecast_features

    features = build_live_forecast_features(
        current_aqi=current_aqi,
        pollutants=pollutants,
        timestamp=timestamp,
    )

    model = joblib.load(MODEL_PATH)

    prediction = float(
        model.predict(features)[0]
    )

    prediction = max(
        0,
        min(500, prediction)
    )

    change = prediction - current_aqi

    print("\n================================")
    print("       AIR AWARE FORECAST")
    print("================================")

    print(
        f"\nLocation: {data['location']}"
    )

    print(
        f"Latest measurement: {data['timestamp']}"
    )

    print(
        f"Data age: {data['age_hours']:.1f} hours"
    )

    print(
        f"\nCurrent estimated AQI: {current_aqi}"
    )

    print(
        f"Current category: "
        f"{get_aqi_category(current_aqi)}"
    )

    print(
        f"\nTomorrow predicted AQI: "
        f"{prediction:.0f}"
    )

    print(
        f"Predicted category: "
        f"{get_aqi_category(prediction)}"
    )

    print(
        f"\nExpected change: "
        f"{change:+.0f} AQI points"
    )

    print("\nPollutant sub-indices:")

    for pollutant, value in subindices.items():
        print(
            f"{pollutant.upper()}: {value}"
        )

    print(
        "\nNOTE:"
        "\nThis is a research forecast using the "
        "latest available station measurement."
    )


if __name__ == "__main__":
    predict_tomorrow()