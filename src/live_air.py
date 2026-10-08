from __future__ import annotations



import os

from datetime import datetime, timezone

from typing import Any



import requests

from dotenv import load_dotenv





# ============================================================

# CONFIG

# ============================================================



load_dotenv()


def get_api_key() -> str | None:
    """
    Resolve the OpenAQ API key for both hosted and local usage.

    Priority:
    1. Streamlit Cloud Secrets
    2. Local environment / .env

    The key is never hard-coded into the project.
    """
    try:
        import streamlit as st

        secret_key = st.secrets.get("OPENAQ_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    env_key = os.getenv("OPENAQ_API_KEY")
    if env_key:
        return env_key.strip()

    return None


API_KEY = get_api_key()

BASE_URL = "https://api.openaq.org/v3"

MAX_STATION_AGE_DAYS = 14
LOCATION_PAGE_SIZE = 1000








# ============================================================

# CPCB-STYLE BREAKPOINTS

# ============================================================



CPCB_BREAKPOINTS = {



    "pm25": [

        (0, 30, 0, 50),

        (31, 60, 51, 100),

        (61, 90, 101, 200),

        (91, 120, 201, 300),

        (121, 250, 301, 400),

        (251, float("inf"), 401, 500),

    ],



    "pm10": [

        (0, 50, 0, 50),

        (51, 100, 51, 100),

        (101, 250, 101, 200),

        (251, 350, 201, 300),

        (351, 430, 301, 400),

        (431, float("inf"), 401, 500),

    ],



    "no2": [

        (0, 40, 0, 50),

        (41, 80, 51, 100),

        (81, 180, 101, 200),

        (181, 280, 201, 300),

        (281, 400, 301, 400),

        (401, float("inf"), 401, 500),

    ],



    "so2": [

        (0, 40, 0, 50),

        (41, 80, 51, 100),

        (81, 380, 101, 200),

        (381, 800, 201, 300),

        (801, 1600, 301, 400),

        (1601, float("inf"), 401, 500),

    ],



    "o3": [

        (0, 50, 0, 50),

        (51, 100, 51, 100),

        (101, 168, 101, 200),

        (169, 208, 201, 300),

        (209, 748, 301, 400),

        (749, float("inf"), 401, 500),

    ],



    "co": [

        (0, 1, 0, 50),

        (1.1, 2, 51, 100),

        (2.1, 10, 101, 200),

        (10.1, 17, 201, 300),

        (17.1, 34, 301, 400),

        (34.1, float("inf"), 401, 500),

    ],

}





# ============================================================

# HTTP SESSION

# ============================================================



_session = requests.Session()



if API_KEY:

    _session.headers.update({

        "X-API-Key": API_KEY

    })





def _get(

    endpoint: str,

    params: dict[str, Any] | None = None,

) -> dict:



    if not API_KEY:
        raise RuntimeError(
            "OpenAQ API key is not configured. "
            "For local use, add OPENAQ_API_KEY to your .env file. "
            "For Streamlit Cloud, add OPENAQ_API_KEY under App Settings → Secrets."
        )

    response = _session.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=20,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "OpenAQ rejected the API key (401 Unauthorized). "
            "Check that OPENAQ_API_KEY is valid and stored correctly."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "OpenAQ denied this request (403 Forbidden). "
            "Check the API key permissions or OpenAQ account."
        )

    response.raise_for_status()

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("OpenAQ returned an invalid response.") from exc





# ============================================================

# HELPERS

# ============================================================



def _normalise_parameter(name: str) -> str:



    name = name.lower().strip()



    aliases = {

        "pm2.5": "pm25",

        "pm25": "pm25",

        "particulate matter pm2.5": "pm25",



        "pm10": "pm10",

        "particulate matter pm10": "pm10",



        "nitrogen dioxide": "no2",

        "no2": "no2",



        "sulfur dioxide": "so2",

        "sulphur dioxide": "so2",

        "so2": "so2",



        "ozone": "o3",

        "o3": "o3",



        "carbon monoxide": "co",

        "co": "co",

    }



    return aliases.get(name, name)





def _parse_datetime(

    value: str | None,

) -> datetime | None:



    if not value:

        return None



    try:

        return datetime.fromisoformat(

            value.replace(

                "Z",

                "+00:00",

            )

        )



    except ValueError:

        return None





def _age_hours(

    timestamp: datetime,

) -> float:



    now = datetime.now(timezone.utc)



    if timestamp.tzinfo is None:

        timestamp = timestamp.replace(

            tzinfo=timezone.utc

        )



    return max(

        0,

        (

            now

            - timestamp.astimezone(

                timezone.utc

            )

        ).total_seconds() / 3600,

    )





# ============================================================

# LOCATION DISCOVERY

# ============================================================



def get_available_locations(

    max_pages: int = 5,

    max_age_days: int = MAX_STATION_AGE_DAYS,

) -> list[dict]:



    """

    Discover recent Indian monitoring stations.



    This endpoint is intentionally kept lightweight.

    Detailed sensor information is fetched only when

    a station is selected.

    """



    locations = []



    for page in range(

        1,

        max_pages + 1,

    ):



        data = _get(

            "/locations",

            params={

                "iso": "IN",

                "monitor": "true",

                "mobile": "false",

                "limit": LOCATION_PAGE_SIZE,

                "page": page,

            },

        )



        results = data.get(

            "results",

            []

        )



        if not results:

            break



        locations.extend(results)



        if len(results) < LOCATION_PAGE_SIZE:

            break





    cutoff = (

        datetime.now(timezone.utc).timestamp()

        - (

            max_age_days

            * 24

            * 60

            * 60

        )

    )





    usable = []



    for location in locations:



        # ----------------------------------------------------

        # LAST DATA

        # ----------------------------------------------------



        datetime_last = (

            location.get("datetimeLast")

            or {}

        )



        last = datetime_last.get(

            "utc"

        )



        timestamp = _parse_datetime(

            last

        )



        if timestamp is None:

            continue



        if timestamp.timestamp() < cutoff:

            continue





        # ----------------------------------------------------

        # COORDINATES

        # ----------------------------------------------------



        coordinates = (

            location.get(

                "coordinates"

            )

            or {}

        )



        latitude = coordinates.get(

            "latitude"

        )



        longitude = coordinates.get(

            "longitude"

        )



        if latitude is None:

            continue



        if longitude is None:

            continue





        # ----------------------------------------------------

        # SAFE PROVIDER

        # ----------------------------------------------------



        provider = (

            location.get("provider")

            or {}

        )



        provider_name = (

            provider.get(

                "name",

                "Unknown",

            )

        )





        # ----------------------------------------------------

        # SAFE COUNTRY

        # ----------------------------------------------------



        country = (

            location.get("country")

            or {}

        )





        # ----------------------------------------------------

        # SAVE LIGHTWEIGHT STATION

        # ----------------------------------------------------



        usable.append(

            {

                "id": location["id"],



                "name": (

                    location.get("name")

                    or f"Station {location['id']}"

                ),



                "locality": (

                    location.get("locality")

                    or ""

                ),



                "city": (

                    location.get("city")

                    or ""

                ),



                "state": (

                    location.get("state")

                    or ""

                ),



                "country": (

                    country.get(

                        "name",

                        "India",

                    )

                ),



                "latitude": latitude,



                "longitude": longitude,



                "timezone": (

                    location.get(

                        "timezone",

                        "Asia/Kolkata",

                    )

                ),



                "provider": provider_name,



                "last_datetime": last,

            }

        )





    # Newest first.

    usable.sort(

        key=lambda x: (

            x["last_datetime"]

            or ""

        ),

        reverse=True,

    )





    # Remove duplicates.

    unique = {}



    for location in usable:



        unique[

            location["id"]

        ] = location





    return list(

        unique.values()

    )





# ============================================================

# FULL LOCATION DETAILS

# ============================================================



def get_location(

    location_id: int,

) -> dict:



    data = _get(

        f"/locations/{location_id}"

    )



    results = data.get(

        "results",

        []

    )



    if not results:

        raise RuntimeError(

            f"Location {location_id} "

            "was not found."

        )



    return results[0]





# ============================================================

# SENSOR MAPPING

# ============================================================



def _sensor_mapping(

    location: dict,

) -> dict[int, tuple[str, str]]:



    mapping = {}



    sensors = (

        location.get("sensors")

        or []

    )



    for sensor in sensors:



        sensor_id = sensor.get(

            "id"

        )



        parameter = (

            sensor.get("parameter")

            or {}

        )



        parameter_name = parameter.get(

            "name"

        )



        unit = parameter.get(

            "units",

            ""

        )



        if (

            sensor_id is None

            or not parameter_name

        ):

            continue



        pollutant = (

            _normalise_parameter(

                parameter_name

            )

        )



        mapping[sensor_id] = (

            pollutant,

            unit.lower(),

        )



    return mapping





# ============================================================

# UNIT CONVERSION

# ============================================================



def _convert_value(

    pollutant: str,

    value: float,

    unit: str,

) -> float:



    unit = unit.lower()





    # Already µg/m³

    if (

        "µg/m³" in unit

        or "ug/m3" in unit

    ):

        return value





    # NO2: ppb → µg/m³

    if (

        pollutant == "no2"

        and "ppb" in unit

    ):

        return value * 1.88





    # SO2: ppb → µg/m³

    if (

        pollutant == "so2"

        and "ppb" in unit

    ):

        return value * 2.62





    # CO: ppb → mg/m³

    if (

        pollutant == "co"

        and "ppb" in unit

    ):

        return value * 0.001145





    # CO already mg/m³

    if (

        pollutant == "co"

        and "mg/m3" in unit

    ):

        return value





    return value





# ============================================================

# LATEST MEASUREMENTS

# ============================================================



def get_latest_measurements(

    location_id: int,

    location: dict | None = None,

) -> dict:



    # --------------------------------------------------------

    # IMPORTANT:

    # The station list is lightweight.

    # Fetch full station metadata here.

    # --------------------------------------------------------



    if (

        location is None

        or not location.get("sensors")

    ):



        location = get_location(

            location_id

        )





    mapping = _sensor_mapping(

        location

    )





    if not mapping:



        raise RuntimeError(

            "No usable sensors were found "

            f"for station {location_id}."

        )





    # --------------------------------------------------------

    # LATEST VALUES

    # --------------------------------------------------------



    data = _get(

        f"/locations/{location_id}/latest",

        params={

            "limit": 100,

        },

    )





    results = data.get(

        "results",

        []

    )





    pollutants = {}



    timestamps = []





    for item in results:



        sensor_id = item.get(

            "sensorsId"

        )



        if sensor_id not in mapping:

            continue





        pollutant, unit = mapping[

            sensor_id

        ]





        if pollutant not in {

            "pm25",

            "pm10",

            "no2",

            "so2",

            "o3",

            "co",

        }:

            continue





        value = item.get(

            "value"

        )



        if value is None:

            continue





        timestamp = _parse_datetime(

            (

                item.get("datetime")

                or {}

            ).get("utc")

        )





        if timestamp:

            timestamps.append(

                timestamp

            )





        pollutants[pollutant] = (

            _convert_value(

                pollutant,

                float(value),

                unit,

            )

        )





    if not timestamps:



        raise RuntimeError(

            "No usable measurements "

            "found for this station."

        )





    latest_timestamp = max(

        timestamps

    )





    # --------------------------------------------------------

    # SAFE METADATA

    # --------------------------------------------------------



    provider = (

        location.get("provider")

        or {}

    )



    country = (

        location.get("country")

        or {}

    )



    coordinates = (

        location.get(

            "coordinates"

        )

        or {}

    )





    return {



        "location_id": location_id,



        "location": (

            location.get("name")

            or f"Station {location_id}"

        ),



        "locality": (

            location.get(

                "locality"

            )

            or ""

        ),



        "city": (

            location.get("city")

            or ""

        ),



        "state": (

            location.get("state")

            or ""

        ),



        "country": (

            country.get(

                "name",

                "India",

            )

        ),



        "provider": (

            provider.get(

                "name",

                "Unknown",

            )

        ),



        "latitude": coordinates.get(

            "latitude"

        ),



        "longitude": coordinates.get(

            "longitude"

        ),



        "timezone": (

            location.get(

                "timezone",

                "Asia/Kolkata",

            )

        ),



        "timestamp": latest_timestamp,



        "age_hours": _age_hours(

            latest_timestamp

        ),



        "pollutants": pollutants,

    }





# ============================================================

# PUBLIC AIR QUALITY FUNCTION

# ============================================================



def get_live_air_quality(

    location_id: int,

    location: dict | None = None,

) -> dict:



    return get_latest_measurements(

        location_id,

        location,

    )





# ============================================================

# FRESHNESS

# ============================================================



def get_freshness_label(

    age_hours: float,

) -> str:



    if age_hours <= 6:

        return "Recent"



    if age_hours <= 24:

        return "Delayed"



    return "Stale"





# ============================================================

# AQI SUB-INDEX

# ============================================================



def calculate_subindex(

    pollutant: str,

    concentration: float,

) -> float | None:



    breakpoints = CPCB_BREAKPOINTS.get(

        pollutant

    )



    if not breakpoints:

        return None





    for (

        c_low,

        c_high,

        i_low,

        i_high,

    ) in breakpoints:



        if (

            c_low

            <= concentration

            <= c_high

        ):



            if c_high == float(

                "inf"

            ):

                return float(

                    i_low

                )



            return (

                i_low

                + (

                    (

                        concentration

                        - c_low

                    )

                    * (

                        i_high

                        - i_low

                    )

                    / (

                        c_high

                        - c_low

                    )

                )

            )





    return None





# ============================================================

# ESTIMATED CPCB-STYLE AQI

# ============================================================



def estimate_cpcb_style_aqi(

    pollutants: dict,

) -> dict:



    subindices = {}





    for pollutant, value in pollutants.items():



        if value is None:

            continue





        index = calculate_subindex(

            pollutant,

            float(value),

        )





        if index is not None:



            subindices[pollutant] = round(

                index,

                1,

            )





    if not subindices:



        return {

            "aqi": None,

            "subindices": {},

            "normalized": {},

        }





    # Highest available pollutant sub-index.

    aqi = max(

        subindices.values()

    )





    return {

        "aqi": round(aqi),

        "subindices": subindices,

        "normalized": pollutants,

    }





# ============================================================

# AQI CATEGORY

# ============================================================



def get_aqi_category(

    aqi: float | int | None,

) -> str:



    if aqi is None:

        return "Unavailable"



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





# ============================================================

# TEST

# ============================================================



if __name__ == "__main__":



    print("=" * 60)

    print("AIR AWARE INDIA")

    print("OpenAQ Indian Station Discovery Test")

    print("=" * 60)



    print(

        "\nDiscovering recent Indian "

        "monitoring stations..."

    )



    try:



        stations = get_available_locations()



        print(

            f"\nFound {len(stations)} "

            "usable recent stations."

        )



        if not stations:



            print(

                "\nNo recent stations found."

            )



        else:



            print(

                "\nTop available stations:"

            )



            for station in stations[:10]:



                print(

                    f"- {station['name']}"

                    f" | {station['locality']}"

                    f" | ID: {station['id']}"

                )





            station = stations[0]



            print(

                "\nTesting first station:"

            )



            print(

                f"  {station['name']}"

            )





            data = get_live_air_quality(

                station["id"],

                station,

            )





            aqi_result = (

                estimate_cpcb_style_aqi(

                    data["pollutants"]

                )

            )





            print(

                f"\nLocation: "

                f"{data['location']}"

            )



            print(

                f"Latest: "

                f"{data['timestamp']}"

            )



            print(

                f"Age: "

                f"{data['age_hours']:.1f} hours"

            )



            print(

                f"AQI: "

                f"{aqi_result['aqi']}"

            )



            print(

                "Category: "

                f"{get_aqi_category(aqi_result['aqi'])}"

            )





            print("\nPollutants:")



            for key, value in (

                data["pollutants"].items()

            ):



                print(

                    f"  {key}: {value}"

                )





            print("\nSub-indices:")



            for key, value in (

                aqi_result["subindices"].items()

            ):



                print(

                    f"  {key}: {value}"

                )





    except Exception as error:



        print("\nERROR:")



        print(str(error))