import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


from live_air import (
    get_available_locations,
    get_live_air_quality,
    estimate_cpcb_style_aqi,
    get_aqi_category,
    get_freshness_label,
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AirAware India",
    page_icon="🌍",
    layout="wide",
)


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #0b0f14;
    }

    [data-testid="stSidebar"] {
        background: #0d1319;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        background: #111820;
        border: 1px solid #26313d;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 34px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# LOAD HISTORICAL DATA
# ==========================================================

@st.cache_data
def load_historical_data():

    path = BASE_DIR / "data" / "ml_data.csv"

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ==========================================================
# LOAD HISTORICAL MODEL
# ==========================================================

@st.cache_resource
def load_historical_model():

    path = (
        BASE_DIR
        / "models"
        / "airaware_random_forest_v3.joblib"
    )

    return joblib.load(path)


# ==========================================================
# LOAD LIVE MODEL
# ==========================================================

@st.cache_resource
def load_live_model():

    path = (
        BASE_DIR
        / "models"
        / "airaware_live_rf.joblib"
    )

    return joblib.load(path)


# ==========================================================
# LOAD AVAILABLE INDIAN MONITORING STATIONS
# ==========================================================

@st.cache_data(ttl=600)
def load_locations():

    return get_available_locations()


# ==========================================================
# LOAD SELECTED STATION
# ==========================================================

@st.cache_data(ttl=300)
def load_location_air(
    location_id,
    location_data,
):

    data = get_live_air_quality(
        location_id,
        location_data,
    )

    aqi_result = estimate_cpcb_style_aqi(
        data["pollutants"]
    )

    return {
        "location_id": data["location_id"],
        "location": data["location"],
        "locality": data["locality"],
        "provider": data["provider"],
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "timestamp": data["timestamp"],
        "age_hours": data["age_hours"],
        "freshness": get_freshness_label(
            data["age_hours"]
        ),
        "pollutants": data["pollutants"],
        "aqi": aqi_result["aqi"],
        "category": get_aqi_category(
            aqi_result["aqi"]
        ),
        "subindices": aqi_result["subindices"],
    }


# ==========================================================
# HISTORICAL AQI CATEGORY
# ==========================================================

def historical_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Satisfactory"

    if aqi <= 200:
        return "Moderate"

    if aqi <= 300:
        return "Poor"

    if aqi <= 400:
        return "Very Poor"

    return "Severe"


# ==========================================================
# FORMAT DATA AGE
# ==========================================================

def format_age(hours):

    if hours < 1:
        return f"{hours * 60:.0f} minutes ago"

    if hours < 24:
        return f"{hours:.1f} hours ago"

    return f"{hours / 24:.1f} days ago"


# ==========================================================
# LOAD MAIN DATA
# ==========================================================

df = load_historical_data()

historical_model = load_historical_model()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌍 AirAware India")

st.sidebar.caption(
    "Environmental ML research platform"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigate",
    [
        "Latest Air",
        "Historical Research",
        "Model Performance",
    ],
)


# ==========================================================
# STATION SELECTOR
# ==========================================================

selected_location = None

if page == "Latest Air":

    try:

        available_locations = load_locations()

        if available_locations:

            location_options = {}

            for location in available_locations:

                name = location.get(
                    "name",
                    f"Station {location['id']}",
                )

                locality = location.get(
                    "locality",
                    "",
                )

                if locality:

                    label = (
                        f"{name} — {locality}"
                    )

                else:

                    label = name

                # Prevent duplicate labels

                if label in location_options:

                    label = (
                        f"{label} "
                        f"(ID {location['id']})"
                    )

                location_options[label] = location


            selected_label = st.sidebar.selectbox(
                "📍 Monitoring Station",
                list(location_options.keys()),
            )


            selected_location = location_options[
                selected_label
            ]


            st.sidebar.caption(
                f"{len(available_locations)} "
                "recent stations available"
            )

        else:

            st.sidebar.warning(
                "No recent monitoring stations found."
            )


    except Exception as error:

        st.sidebar.error(
            "Unable to load monitoring stations."
        )

        st.sidebar.caption(
            str(error)
        )


st.sidebar.divider()

st.sidebar.caption(
    "Research Prototype"
)


# ==========================================================
# HEADER
# ==========================================================

st.title("🌍 AirAware India")

st.caption(
    "Machine learning for short-term air-quality "
    "forecasting and understandable environmental insights."
)


# ==========================================================
# LATEST AIR
# ==========================================================

if page == "Latest Air":

    st.header("Latest Available Air Quality")

    st.caption(
        "Latest available monitoring measurement and "
        "AirAware short-term forecast."
    )


    # ------------------------------------------------------
    # CHECK STATION
    # ------------------------------------------------------

    if selected_location is None:

        st.error(
            "No monitoring station is currently available."
        )

        st.stop()


    # ------------------------------------------------------
    # LOAD SELECTED STATION
    # ------------------------------------------------------

    try:

        live = load_location_air(
            selected_location["id"],
            selected_location,
        )

    except Exception as error:

        st.error(
            "Unable to retrieve monitoring data."
        )

        st.code(str(error))

        st.stop()


    # ------------------------------------------------------
    # LOCATION
    # ------------------------------------------------------

    location_text = live["location"]

    if live["locality"]:

        location_text += (
            f" — {live['locality']}"
        )


    st.info(
        f"📍 **{location_text}**  \n"
        f"Provider: **{live['provider']}**  \n"
        f"Latest measurement: "
        f"**{live['timestamp'].strftime('%d %b %Y, %H:%M UTC')}**"
    )


    # ------------------------------------------------------
    # FRESHNESS
    # ------------------------------------------------------

    age = live["age_hours"]


    if age > 24:

        st.warning(
            f"⚠️ **Data is delayed:** "
            f"the latest available measurement is "
            f"**{format_age(age)}**. "
            f"This is not presented as real-time "
            f"official AQI."
        )

    else:

        st.success(
            f"Monitoring data status: "
            f"**{live['freshness']}**"
        )


    # ------------------------------------------------------
    # CURRENT AIR QUALITY
    # ------------------------------------------------------

    current_aqi = live["aqi"]

    pollutants = live["pollutants"]


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Current Air Quality")

        st.metric(
            "Estimated AQI",
            (
                current_aqi
                if current_aqi is not None
                else "—"
            ),
        )

        if current_aqi is not None:

            st.caption(
                f"Category: **{live['category']}**"
            )


    # ------------------------------------------------------
    # TOMORROW FORECAST
    # ------------------------------------------------------

    tomorrow_prediction = None


    if current_aqi is not None:

        timestamp = live["timestamp"]

        forecast_features = pd.DataFrame(
            [
                {
                    "AQI": current_aqi,

                    "PM2.5": pollutants.get(
                        "pm25",
                        0,
                    ),

                    "PM10": pollutants.get(
                        "pm10",
                        0,
                    ),

                    "Day": timestamp.day,

                    "Month": timestamp.month,

                    "DayOfWeek": timestamp.weekday(),
                }
            ]
        )


        try:

            live_model = load_live_model()


            tomorrow_prediction = float(
                live_model.predict(
                    forecast_features
                )[0]
            )


            tomorrow_prediction = round(
                max(
                    0,
                    min(
                        tomorrow_prediction,
                        500,
                    ),
                )
            )


        except Exception as error:

            st.warning(
                f"Forecast unavailable: {error}"
            )


    with col2:

        st.subheader("Tomorrow Forecast")


        if tomorrow_prediction is not None:

            change = (
                tomorrow_prediction
                - current_aqi
            )


            st.metric(
                "Predicted AQI",
                tomorrow_prediction,
                delta=f"{change:+.0f}",
            )


            st.caption(
                f"Category: "
                f"**{get_aqi_category(tomorrow_prediction)}**"
            )


        else:

            st.metric(
                "Predicted AQI",
                "—",
            )


    # ======================================================
    # POLLUTANTS
    # ======================================================

    st.divider()

    st.header("Latest Pollutant Measurements")

    st.caption(
        "Measurements reported by the selected "
        "monitoring station."
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(
            "PM2.5",
            (
                f"{pollutants['pm25']:.2f}"
                if pollutants.get("pm25") is not None
                else "—"
            ),
            help="µg/m³",
        )


    with p2:

        st.metric(
            "PM10",
            (
                f"{pollutants['pm10']:.2f}"
                if pollutants.get("pm10") is not None
                else "—"
            ),
            help="µg/m³",
        )


    with p3:

        st.metric(
            "NO₂",
            (
                f"{pollutants['no2']:.2f}"
                if pollutants.get("no2") is not None
                else "—"
            ),
            help="µg/m³",
        )


    p4, p5, p6 = st.columns(3)


    with p4:

        st.metric(
            "SO₂",
            (
                f"{pollutants['so2']:.2f}"
                if pollutants.get("so2") is not None
                else "—"
            ),
            help="µg/m³",
        )


    with p5:

        st.metric(
            "CO",
            (
                f"{pollutants['co']:.2f}"
                if pollutants.get("co") is not None
                else "—"
            ),
            help="mg/m³",
        )


    with p6:

        st.metric(
            "O₃",
            (
                f"{pollutants['o3']:.2f}"
                if pollutants.get("o3") is not None
                else "—"
            ),
            help="µg/m³",
        )


    # ======================================================
    # SUBINDICES
    # ======================================================

    st.divider()

    st.header("AQI Pollutant Contributions")

    st.caption(
        "Estimated CPCB-style pollutant sub-indices. "
        "The highest available sub-index determines "
        "the displayed estimated AQI."
    )


    subindices = live["subindices"]


    if subindices:

        subindex_df = pd.DataFrame(
            {
                "Pollutant": [
                    key.upper()
                    for key in subindices
                ],

                "Estimated Sub-index": [
                    round(
                        value,
                        1,
                    )
                    for value in subindices.values()
                ],
            }
        )


        st.dataframe(
            subindex_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No pollutant sub-indices available."
        )


    # ======================================================
    # AIRWARE INSIGHT & GUIDANCE
    # ======================================================

    st.divider()

    st.header("AirAware Insight")


    if current_aqi is not None:


        # --------------------------------------------------
        # CATEGORY-SPECIFIC MESSAGE
        # --------------------------------------------------

        if current_aqi <= 50:

            insight = "Air quality is Good."

            guidance = (
                "Air pollution levels are low. "
                "Normal outdoor activities can continue."
            )


        elif current_aqi <= 100:

            insight = "Air quality is Satisfactory."

            guidance = (
                "Air quality is generally acceptable. "
                "Most people can continue normal outdoor activities."
            )


        elif current_aqi <= 200:

            insight = "Air quality is Moderate."

            guidance = (
                "Air quality may be less comfortable for some people. "
                "Consider reducing prolonged outdoor activity if you "
                "notice discomfort."
            )


        elif current_aqi <= 300:

            insight = "Air quality is Poor."

            guidance = (
                "Pollution levels are high. Consider reducing prolonged "
                "outdoor activity, particularly if you are sensitive "
                "to air pollution."
            )


        elif current_aqi <= 400:

            insight = "Air quality is Very Poor."

            guidance = (
                "Pollution levels are very high. Consider limiting "
                "prolonged outdoor exposure and paying attention to "
                "air-quality conditions before spending extended time outside."
            )


        else:

            insight = "Air quality is Severe."

            guidance = (
                "Pollution levels are extremely high. Consider avoiding "
                "prolonged outdoor exposure and closely monitoring "
                "local air-quality conditions."
            )


        # --------------------------------------------------
        # MAIN INSIGHT
        # --------------------------------------------------

        st.info(
            f"**{insight}**\n\n"
            f"{guidance}"
        )


        # --------------------------------------------------
        # TOMORROW'S OUTLOOK
        # --------------------------------------------------

        if tomorrow_prediction is not None:

            tomorrow_category = get_aqi_category(
                tomorrow_prediction
            )

            change = (
                tomorrow_prediction
                - current_aqi
            )


            if change < -5:

                outlook = (
                    f"AirAware forecasts AQI to decrease from "
                    f"**{current_aqi:.0f} → {tomorrow_prediction:.0f}** "
                    f"tomorrow. Air quality is still expected to be "
                    f"**{tomorrow_category}**."
                )


            elif change > 5:

                outlook = (
                    f"AirAware forecasts AQI to increase from "
                    f"**{current_aqi:.0f} → {tomorrow_prediction:.0f}** "
                    f"tomorrow. Air quality is expected to remain "
                    f"**{tomorrow_category}**."
                )


            else:

                outlook = (
                    f"AirAware forecasts AQI to remain broadly similar "
                    f"at around **{tomorrow_prediction:.0f}** tomorrow, "
                    f"with air quality expected to be "
                    f"**{tomorrow_category}**."
                )


            st.success(
                f"**Tomorrow's outlook:** {outlook}"
            )


        # --------------------------------------------------
        # SIMPLE GUIDANCE
        # --------------------------------------------------

        st.subheader("What You Can Do")

        if current_aqi <= 50:

            st.write(
                "• Normal outdoor activities are appropriate."
            )

        elif current_aqi <= 100:

            st.write(
                "• Most people can continue normal outdoor activities."
            )

        elif current_aqi <= 200:

            st.write(
                "• Consider reducing prolonged outdoor activity "
                "if you experience discomfort."
            )

        elif current_aqi <= 300:

            st.write(
                "• Consider reducing prolonged outdoor activity."
            )

            st.write(
                "• People sensitive to air pollution should "
                "take extra care with extended outdoor exposure."
            )

        elif current_aqi <= 400:

            st.write(
                "• Consider limiting prolonged outdoor exposure."
            )

            st.write(
                "• Check air-quality conditions before spending "
                "extended periods outdoors."
            )

        else:

            st.write(
                "• Consider avoiding prolonged outdoor exposure."
            )

            st.write(
                "• Closely monitor local air-quality conditions."
            )


        # --------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------

        st.caption(
            "AirAware provides environmental information based on "
            "available monitoring data and a research forecasting model. "
            "It is not medical advice."
        )


    else:

        st.info(
            "Air quality insight is unavailable because "
            "a current AQI estimate could not be calculated."
        )


    # ======================================================
    # DATA & RESEARCH NOTE
    # ======================================================

    st.divider()

    st.header("Data & Research Note")

    st.markdown(
        """
        **Data source:** OpenAQ monitoring data.

        **Forecast:** AirAware Random Forest model trained
        on historical Indian air-quality observations.

        **Important limitation:** Official CPCB AQI uses
        prescribed averaging periods and sufficient observations.
        AirAware's current estimate is calculated from the
        latest available station measurements and is therefore
        labelled an **estimated CPCB-style AQI**.

        The current monitoring data may also be delayed.
        """
    )


# ==========================================================
# HISTORICAL RESEARCH
# ==========================================================

elif page == "Historical Research":

    st.header("Historical Air Quality Research")

    st.caption(
        "Explore the historical Indian air-quality dataset "
        "and next-day AQI forecasting model."
    )


    st.info(
        "Historical research mode • 2015–2020 dataset"
    )


    # ------------------------------------------------------
    # CONTROLS
    # ------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        cities = sorted(
            df["City"].unique()
        )

        city = st.selectbox(
            "📍 Select city",
            cities,
        )


    with col2:

        city_data = df[
            df["City"] == city
        ]

        dates = sorted(
            city_data["Date"]
            .dt
            .date
            .unique()
        )

        date = st.selectbox(
            "📅 Select date",
            dates,
        )


    # ------------------------------------------------------
    # RECORD
    # ------------------------------------------------------

    selected_rows = city_data[
        city_data["Date"].dt.date == date
    ]


    if selected_rows.empty:

        st.warning(
            "No data available for this date."
        )

        st.stop()


    selected = selected_rows.iloc[0]


    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    features = [
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


    prediction = historical_model.predict(
        pd.DataFrame(
            [selected[features]]
        )
    )[0]


    # ------------------------------------------------------
    # ACTUAL NEXT DAY
    # ------------------------------------------------------

    next_date = (
        pd.Timestamp(date)
        + pd.Timedelta(days=1)
    )


    actual_row = city_data[
        city_data["Date"] == next_date
    ]


    actual_next_day = None


    if not actual_row.empty:

        actual_next_day = (
            actual_row.iloc[0]["AQI"]
        )


    prediction_error = None


    if actual_next_day is not None:

        prediction_error = abs(
            prediction
            - actual_next_day
        )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    st.header("Air Quality Overview")


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Today's AQI",
            f"{selected['AQI']:.0f}",
        )

        st.caption(
            historical_aqi_category(
                selected["AQI"]
            )
        )


    with c2:

        st.metric(
            "Predicted Tomorrow",
            f"{prediction:.0f}",
        )

        st.caption(
            historical_aqi_category(
                prediction
            )
        )


    with c3:

        st.metric(
            "AQI Change",
            f"{selected['AQI_Change']:+.0f}",
        )


    with c4:

        if actual_next_day is not None:

            st.metric(
                "Actual Tomorrow",
                f"{actual_next_day:.0f}",
            )

        else:

            st.metric(
                "Actual Tomorrow",
                "—",
            )


    # ------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------

    st.header("Forecast Evaluation")


    if actual_next_day is not None:

        st.success(
            f"AirAware predicted **{prediction:.0f} AQI**. "
            f"The actual next-day AQI was "
            f"**{actual_next_day:.0f} AQI**. "
            f"Prediction error: "
            f"**{prediction_error:.1f} AQI points**."
        )

    else:

        st.info(
            "Historical next-day AQI is not available "
            "for this date."
        )


    # ------------------------------------------------------
    # POLLUTANTS
    # ------------------------------------------------------

    st.header("Pollution Indicators")


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "PM2.5",
            f"{selected['PM2.5']:.2f}",
        )


    with c2:

        st.metric(
            "PM10",
            f"{selected['PM10']:.2f}",
        )


    with c3:

        st.metric(
            "NO₂",
            f"{selected['NO2']:.2f}",
        )


    c4, c5, c6 = st.columns(3)


    with c4:

        st.metric(
            "CO",
            f"{selected['CO']:.2f}",
        )


    with c5:

        st.metric(
            "SO₂",
            f"{selected['SO2']:.2f}",
        )


    with c6:

        st.metric(
            "O₃",
            f"{selected['O3']:.2f}",
        )


    # ------------------------------------------------------
    # TREND
    # ------------------------------------------------------

    st.header("Recent AQI Trend")


    history = (
        city_data[
            city_data["Date"]
            <= pd.Timestamp(date)
        ]
        .tail(7)[
            ["Date", "AQI"]
        ]
    )


    history = history.set_index(
        "Date"
    )


    st.line_chart(history)


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.header("AirAware Model Performance")

    st.caption(
        "Evaluation of the historical next-day AQI "
        "forecasting model."
    )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Model",
            "Random Forest",
        )


    with c2:

        st.metric(
            "Test MAE",
            "16.20",
        )


    with c3:

        st.metric(
            "Test R²",
            "0.883",
        )


    # ------------------------------------------------------
    # EXPLANATION
    # ------------------------------------------------------

    st.header("What the Metrics Mean")


    st.markdown(
        """
        **MAE — 16.20 AQI points**

        On average, the model's prediction differs from
        the actual next-day AQI by about 16 AQI points on
        the held-out test set.

        **R² — 0.883**

        The model explains a large proportion of the variation
        in next-day AQI within this test set.

        These values describe performance on the historical
        test data. They should **not** be interpreted as
        "88.3% accuracy."
        """
    )


    # ------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------

    st.header("Model Inputs")


    feature_list = [
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


    st.dataframe(
        pd.DataFrame(
            {
                "Feature": feature_list
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ------------------------------------------------------
    # LIMITATIONS
    # ------------------------------------------------------

    st.header("Research Limitations")


    st.warning(
        """
        AirAware is currently a research prototype.

        The historical forecasting model was trained using
        Indian city-level air-quality observations.

        The latest-monitoring pipeline currently uses OpenAQ
        station measurements. These measurements may be delayed
        and are not necessarily equivalent to the averaging
        periods used for official CPCB AQI.

        Therefore AirAware should be viewed as an experimental
        forecasting and environmental information system rather
        than an official government AQI service.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "AirAware India • ML + Environmental Data Research Prototype"
)