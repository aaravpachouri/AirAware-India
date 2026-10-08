import sys

from pathlib import Path

from html import escape



import joblib

import pandas as pd

import streamlit as st

import streamlit.components.v1 as components





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

    initial_sidebar_state="expanded",

)





# ==========================================================

# PREMIUM GLOBAL CSS

# ==========================================================



st.markdown(

    """

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');



:root {

    --aa-bg: #05080d;

    --aa-panel: rgba(13, 19, 27, 0.72);

    --aa-panel-strong: rgba(13, 20, 29, 0.92);

    --aa-border: rgba(255,255,255,0.075);

    --aa-text: #f4f7fb;

    --aa-muted: #8795a6;

    --aa-accent: #55e2b0;

    --aa-blue: #5aa8ff;

}



html, body, [class*="css"] {

    font-family: "Inter", sans-serif !important;

}



.stApp {

    background:

        radial-gradient(circle at 8% 8%, rgba(85,226,176,.065), transparent 24%),

        radial-gradient(circle at 94% 10%, rgba(90,168,255,.065), transparent 24%),

        linear-gradient(135deg, #05080d 0%, #07101a 52%, #04070b 100%);

    color: var(--aa-text);

}



.block-container {

    max-width: 1460px;

    padding-top: 2rem;

    padding-bottom: 5rem;

}



/* AirAware replacement for Streamlit Material Symbols sidebar icon */
[data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"] {
    font-size: 0 !important;
    line-height: 0 !important;
}
[data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"]::after {
    content: "»";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    color: #8fa0b5;
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 25px !important;
    font-weight: 700;
    line-height: 1;
    transition: transform .22s ease, color .22s ease, text-shadow .22s ease;
}
[data-testid="stSidebarCollapseButton"] button:hover [data-testid="stIconMaterial"]::after {
    color: #ffffff;
    transform: translateX(3px);
    text-shadow: 0 0 9px rgba(85,226,176,.55), 0 0 18px rgba(85,226,176,.2);
}

[data-testid="stSidebar"] {

    background:

        linear-gradient(180deg, rgba(6,10,15,.97), rgba(4,8,12,.99)) !important;

    border-right: 1px solid rgba(255,255,255,.06);

}



[data-testid="stSidebar"] * {

    font-family: "Inter", sans-serif !important;

}



h1, h2, h3 {

    color: var(--aa-text) !important;

    letter-spacing: -.04em !important;

}



h1 {

    font-size: 2.55rem !important;

    font-weight: 800 !important;

}



h2 {

    font-size: 2rem !important;

    font-weight: 750 !important;

}



h3 {

    font-size: 1.22rem !important;

    font-weight: 700 !important;

}



p, label, .stCaption {

    color: var(--aa-muted) !important;

}



div[data-testid="stMetric"] {

    background:

        linear-gradient(145deg, rgba(17,26,36,.94), rgba(8,14,21,.9));

    border: 1px solid var(--aa-border);

    border-radius: 20px;

    padding: 21px;

    box-shadow:

        0 20px 60px rgba(0,0,0,.22),

        inset 0 1px 0 rgba(255,255,255,.025);

    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;

}



div[data-testid="stMetric"]:hover {

    transform: translateY(-4px);

    border-color: rgba(85,226,176,.25);

    box-shadow:

        0 24px 68px rgba(0,0,0,.32),

        0 0 28px rgba(85,226,176,.05);

}



div[data-testid="stMetricLabel"] {

    color: #8e9cac !important;

    font-size: .82rem !important;

    font-weight: 600 !important;

}



div[data-testid="stMetricValue"] {

    color: #f2f6fa !important;

    font-size: 2.08rem !important;

    font-weight: 800 !important;

}



.stSelectbox > div > div,

.stMultiSelect > div > div,

.stTextInput > div > div {

    background: rgba(9,15,22,.88) !important;

    border-radius: 13px !important;

    border-color: rgba(255,255,255,.08) !important;

}



.stButton > button {

    border: 1px solid rgba(255,255,255,.08);

    border-radius: 12px;

    background: linear-gradient(135deg, rgba(24,35,47,.96), rgba(10,16,23,.96));

    color: #eff4f8;

    font-weight: 650;

    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;

}



.stButton > button:hover {

    transform: translateY(-2px);

    border-color: rgba(85,226,176,.28);

    box-shadow: 0 10px 28px rgba(0,0,0,.28);

}



div[data-testid="stAlert"] {

    border-radius: 16px;

    border: 1px solid rgba(255,255,255,.07);

    background: rgba(12,18,25,.72);

    backdrop-filter: blur(18px);

}



[data-testid="stDataFrame"] {

    border-radius: 16px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,.07);

}



hr {

    border-color: rgba(255,255,255,.06) !important;

}



[data-testid="stDecoration"] {

    display: none;

}



header {

    background: transparent !important;

}



::-webkit-scrollbar { width: 8px; }

::-webkit-scrollbar-track { background: #05080d; }

::-webkit-scrollbar-thumb {

    background: #1d2935;

    border-radius: 999px;

}

::-webkit-scrollbar-thumb:hover { background: #2c3e4f; }



/* =========================================================

   AIR AWARE // APP MOTION SYSTEM

   ========================================================= */



html {

    scroll-behavior: smooth;

}



body::before {

    content: "";

    position: fixed;

    top: 0;

    left: 0;

    width: 100%;

    height: 2px;

    z-index: 999999;

    pointer-events: none;

    transform-origin: left center;

    background: linear-gradient(90deg, #55e2b0, #5aa8ff, #a683ff);

    transform: scaleX(0);

    animation: aaScrollProgress linear both;

    animation-timeline: scroll(root block);

}



@keyframes aaScrollProgress {

    to { transform: scaleX(1); }

}



/* Smooth viewport entry for our custom section shells. */

.aa-reveal {

    opacity: 0;

    transform: translateY(24px) scale(.985);

    filter: blur(3px);

}



@supports (animation-timeline: view()) {

    .aa-reveal {

        animation: aaViewReveal linear both;

        animation-timeline: view();

        animation-range: entry 6% cover 30%;

    }

}



@keyframes aaViewReveal {

    to {

        opacity: 1;

        transform: translateY(0) scale(1);

        filter: blur(0);

    }

}



/* Sidebar entrance + motion-rich navigation. */

[data-testid="stSidebar"] {

    animation: aaSidebarEntrance .85s cubic-bezier(.16,1,.3,1) both;

}



@keyframes aaSidebarEntrance {

    from {

        opacity: 0;

        transform: translateX(-22px);

    }

    to {

        opacity: 1;

        transform: translateX(0);

    }

}



[data-testid="stSidebar"] div[role="radiogroup"] {

    gap: 3px;

}



[data-testid="stSidebar"] div[role="radiogroup"] label {

    position: relative;

    overflow: hidden;

    border-radius: 13px !important;

    padding: 8px 10px !important;

    margin: 2px 0 !important;

    transition:

        transform .24s cubic-bezier(.16,1,.3,1),

        background .24s ease,

        box-shadow .24s ease,

        color .24s ease;

}



[data-testid="stSidebar"] div[role="radiogroup"] label::after {

    content: "";

    position: absolute;

    inset: 0 auto 0 -100%;

    width: 100%;

    background: linear-gradient(

        90deg,

        transparent,

        rgba(85,226,176,.08),

        transparent

    );

    transition: left .5s ease;

    pointer-events: none;

}



[data-testid="stSidebar"] div[role="radiogroup"] label:hover {

    transform: translateX(6px);

    background: rgba(85,226,176,.05) !important;

    box-shadow:

        inset 2px 0 0 rgba(85,226,176,.45),

        0 8px 24px rgba(0,0,0,.12);

}



[data-testid="stSidebar"] div[role="radiogroup"] label:hover::after {

    left: 100%;

}



[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {

    background:

        linear-gradient(

            90deg,

            rgba(85,226,176,.105),

            rgba(85,226,176,.018)

        ) !important;

    box-shadow:

        inset 2px 0 0 rgba(85,226,176,.9),

        0 8px 28px rgba(0,0,0,.14);

}



[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {

    animation: aaBrandEntrance 1s cubic-bezier(.16,1,.3,1) both;

}



@keyframes aaBrandEntrance {

    from {

        opacity: 0;

        transform: translateY(-10px);

        filter: blur(4px);

    }

    to {

        opacity: 1;

        transform: translateY(0);

        filter: blur(0);

    }

}



/* Inputs get a subtle lift rather than an abrupt state change. */

.stSelectbox > div > div,

.stMultiSelect > div > div,

.stTextInput > div > div {

    transition:

        border-color .22s ease,

        box-shadow .22s ease,

        transform .22s ease;

}



.stSelectbox > div > div:hover,

.stMultiSelect > div > div:hover,

.stTextInput > div > div:hover {

    transform: translateY(-1px);

    border-color: rgba(85,226,176,.18) !important;

    box-shadow: 0 8px 26px rgba(0,0,0,.16);

}



/* Reduced-motion accessibility. */

@media (prefers-reduced-motion: reduce) {

    html { scroll-behavior: auto; }



    *,

    *::before,

    *::after {

        animation-duration: .001ms !important;

        animation-iteration-count: 1 !important;

        transition-duration: .001ms !important;

    }



    body::before {

        display: none;

    }

}



.aa-page-chip {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 8px 12px;

    border: 1px solid rgba(255,255,255,.07);

    border-radius: 999px;

    background: rgba(255,255,255,.035);

    color: #9eabb9;

    font-size: .72rem;

    letter-spacing: .11em;

    text-transform: uppercase;

    font-weight: 700;

}



.aa-page-chip .dot {

    width: 7px;

    height: 7px;

    border-radius: 50%;

    background: var(--aa-accent);

    box-shadow: 0 0 14px rgba(85,226,176,.75);

}



.aa-section-title {

    margin-top: .45rem;

    margin-bottom: .1rem;

}



.aa-section-subtitle {

    color: #7f8d9d;

    margin-bottom: 1rem;

}



.aa-note {

    padding: 16px 18px;

    border-radius: 15px;

    background: rgba(255,255,255,.026);

    border: 1px solid rgba(255,255,255,.06);

    color: #9daab8;

}



.aa-insight {

    padding: 21px 23px;

    border-radius: 19px;

    background:

        linear-gradient(135deg, rgba(85,226,176,.075), rgba(90,168,255,.035));

    border: 1px solid rgba(85,226,176,.12);

}



@media (max-width: 900px) {

    .block-container {

        padding-left: 1rem;

        padding-right: 1rem;

    }

}

</style>

""",

    unsafe_allow_html=True,

)





# ==========================================================

# DATA / MODEL LOADERS

# ==========================================================



@st.cache_data

def load_historical_data():

    path = BASE_DIR / "data" / "ml_data.csv"

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    return df





@st.cache_resource

def load_historical_model():

    path = BASE_DIR / "models" / "airaware_random_forest_v3.joblib"

    return joblib.load(path)





@st.cache_resource

def load_live_model():

    path = BASE_DIR / "models" / "airaware_live_rf.joblib"

    return joblib.load(path)





@st.cache_data(ttl=600)

def load_locations():

    return get_available_locations()





@st.cache_data(ttl=300)

def load_location_air(location_id, location_data):

    data = get_live_air_quality(location_id, location_data)

    aqi_result = estimate_cpcb_style_aqi(data["pollutants"])



    return {

        "location_id": data["location_id"],

        "location": data["location"],

        "locality": data["locality"],

        "provider": data["provider"],

        "latitude": data["latitude"],

        "longitude": data["longitude"],

        "timestamp": data["timestamp"],

        "age_hours": data["age_hours"],

        "freshness": get_freshness_label(data["age_hours"]),

        "pollutants": data["pollutants"],

        "aqi": aqi_result["aqi"],

        "category": get_aqi_category(aqi_result["aqi"]),

        "subindices": aqi_result["subindices"],

    }





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





def format_age(hours):

    if hours < 1:

        return f"{hours * 60:.0f} minutes ago"

    if hours < 24:

        return f"{hours:.1f} hours ago"

    return f"{hours / 24:.1f} days ago"





def safe_float(value):

    try:

        if value is None or pd.isna(value):

            return None

        return float(value)

    except (TypeError, ValueError):

        return None





# ==========================================================

# HIGH-END ANIMATED HERO

# GSAP + Three.js loaded from official CDN packages.

# ==========================================================



def render_animated_hero(

    current_aqi,

    category,

    location_text,

    freshness,

    tomorrow_prediction,

    provider,

):

    """

    High-end browser hero.



    Essential data is rendered server-side so the dashboard never depends

    on an animation CDN for correctness. GSAP and Three.js enhance the

    experience when the browser can load them.

    """



    actual_aqi = None

    if current_aqi is not None:

        try:

            actual_aqi = max(0.0, min(float(current_aqi), 500.0))

        except (TypeError, ValueError):

            actual_aqi = None



    if actual_aqi is None:

        aqi_display = "—"

        progress = 0

    else:

        aqi_display = f"{actual_aqi:.0f}"

        progress = actual_aqi / 500



    tomorrow = None

    if tomorrow_prediction is not None:

        try:

            tomorrow = max(0, min(int(round(float(tomorrow_prediction))), 500))

        except (TypeError, ValueError):

            tomorrow = None



    if actual_aqi is not None and tomorrow is not None:

        delta = int(tomorrow - round(actual_aqi))



        if delta > 5:

            delta_text = f"↑ {delta:+d}"

            direction_text = "Expected to worsen"

        elif delta < -5:

            delta_text = f"↓ {delta:+d}"

            direction_text = "Expected to improve"

        else:

            delta_text = f"→ {delta:+d}"

            direction_text = "Expected to remain similar"

    else:

        delta_text = "—"

        direction_text = "Forecast unavailable"



    tomorrow_text = "—" if tomorrow is None else str(tomorrow)



    category_key = str(category or "Unavailable").strip().lower()



    accent_map = {

        "good": "#55e2b0",

        "satisfactory": "#63d2ff",

        "moderate": "#f2cf69",

        "poor": "#ff9d67",

        "very poor": "#ff687f",

        "severe": "#b784ff",

        "unavailable": "#7f8b99",

    }



    accent = accent_map.get(

        category_key,

        "#55e2b0",

    )



    safe_location = escape(str(location_text or "Unknown station"))

    safe_provider = escape(str(provider or "Unknown"))

    safe_freshness = escape(str(freshness or "Unknown"))

    safe_category = escape(str(category or "Unavailable"))



    html = f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

* {{

    box-sizing: border-box;

}}



html, body {{

    width: 100%;

    height: 100%;

    margin: 0;

    overflow: hidden;

    background: transparent;

    font-family:

        Inter,

        system-ui,

        -apple-system,

        BlinkMacSystemFont,

        "Segoe UI",

        sans-serif;

}}



.aa-hero {{

    --hero-accent: {accent};



    position: relative;

    width: 100%;

    height: 100%;

    min-height: 560px;

    overflow: hidden;



    border-radius: 30px;

    border: 1px solid rgba(255,255,255,.09);



    background:

        radial-gradient(

            circle at 50% 43%,

            color-mix(

                in srgb,

                var(--hero-accent) 9%,

                transparent

            ),

            transparent 27%

        ),

        radial-gradient(

            circle at 12% 10%,

            rgba(90,168,255,.045),

            transparent 30%

        ),

        linear-gradient(

            135deg,

            rgba(7,13,20,.96),

            rgba(4,9,14,.985)

        );



    box-shadow:

        0 34px 100px rgba(0,0,0,.42),

        inset 0 1px 0 rgba(255,255,255,.04);

}}



#aa-canvas {{

    position: absolute;

    inset: 0;

    width: 100%;

    height: 100%;

    opacity: .82;

    z-index: 0;

    pointer-events: none;

}}



.aa-noise {{

    position: absolute;

    inset: 0;

    z-index: 1;

    pointer-events: none;

    opacity: .16;

    background-image:

        radial-gradient(

            rgba(255,255,255,.055) .6px,

            transparent .7px

        );

    background-size: 4px 4px;

    mix-blend-mode: screen;

}}



.aa-sweep {{

    position: absolute;

    top: -20%;

    left: -35%;

    width: 35%;

    height: 160%;

    z-index: 2;

    pointer-events: none;

    transform: rotate(14deg);

    background: linear-gradient(

        90deg,

        transparent,

        rgba(255,255,255,.035),

        transparent

    );

    animation: aaSweep 7.5s ease-in-out infinite;

}}



@keyframes aaSweep {{

    0%, 70% {{

        transform: translateX(0) rotate(14deg);

        opacity: 0;

    }}

    82% {{

        opacity: .8;

    }}

    100% {{

        transform: translateX(430%) rotate(14deg);

        opacity: 0;

    }}

}}



.aa-hud {{

    position: absolute;

    inset: 0;

    z-index: 5;



    display: grid;

    grid-template-columns: 1fr minmax(300px, 1.2fr) 1fr;

    grid-template-rows: auto 1fr auto;



    padding: 31px;

    gap: 18px;

}}



.aa-top {{

    grid-column: 1 / -1;



    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 18px;

}}



.aa-eyebrow {{

    color: #7e8c9e;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: .20em;

    text-transform: uppercase;

}}



.aa-live-badge {{

    display: inline-flex;

    align-items: center;

    gap: 8px;



    padding: 8px 11px;

    border-radius: 999px;



    color: #9daab8;

    font-size: 9px;

    font-weight: 750;

    letter-spacing: .08em;

    text-transform: uppercase;



    background: rgba(255,255,255,.025);

    border: 1px solid rgba(255,255,255,.065);

    backdrop-filter: blur(14px);

}}



.aa-live-dot {{

    width: 7px;

    height: 7px;

    border-radius: 50%;



    background: var(--hero-accent);

    box-shadow:

        0 0 9px var(--hero-accent),

        0 0 21px color-mix(

            in srgb,

            var(--hero-accent) 55%,

            transparent

        );



    animation: aaBlink 2s ease-in-out infinite;

}}



@keyframes aaBlink {{

    0%, 100% {{

        opacity: 1;

        transform: scale(1);

    }}

    50% {{

        opacity: .42;

        transform: scale(.72);

    }}

}}



.aa-side {{

    display: flex;

    flex-direction: column;

    justify-content: center;

    gap: 14px;

}}



.aa-panel-label {{

    color: #748294;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: .16em;

    text-transform: uppercase;

}}



.aa-panel-value {{

    color: #edf3f8;

    font-size: 19px;

    line-height: 1.2;

    font-weight: 750;

    letter-spacing: -.025em;

}}



.aa-panel-meta {{

    margin-top: -6px;

    color: #7b8a9a;

    font-size: 10px;

    line-height: 1.5;

}}



.aa-side-card {{

    padding: 17px 18px;

    border-radius: 18px;



    background:

        linear-gradient(

            140deg,

            rgba(255,255,255,.042),

            rgba(255,255,255,.018)

        );



    border: 1px solid rgba(255,255,255,.065);

    backdrop-filter: blur(14px);



    box-shadow:

        inset 0 1px 0 rgba(255,255,255,.025),

        0 14px 34px rgba(0,0,0,.15);



    transition:

        transform .25s cubic-bezier(.16,1,.3,1),

        border-color .25s ease,

        background .25s ease;

}}



.aa-side-card:hover {{

    transform: translateY(-4px);

    border-color: color-mix(

        in srgb,

        var(--hero-accent) 30%,

        rgba(255,255,255,.065)

    );

    background: rgba(255,255,255,.052);

}}



.aa-center {{

    position: relative;



    display: flex;

    align-items: center;

    justify-content: center;

}}



.aa-orbit {{

    position: absolute;

    width: 330px;

    height: 330px;

    border: 1px solid rgba(255,255,255,.035);

    border-radius: 50%;

    animation: aaOrbit 18s linear infinite;

}}



.aa-orbit::before,

.aa-orbit::after {{

    content: "";

    position: absolute;

    border-radius: 50%;

    background: var(--hero-accent);

    box-shadow:

        0 0 17px var(--hero-accent),

        0 0 35px color-mix(

            in srgb,

            var(--hero-accent) 42%,

            transparent

        );

}}



.aa-orbit::before {{

    width: 5px;

    height: 5px;

    top: 16px;

    left: 50%;

}}



.aa-orbit::after {{

    width: 3px;

    height: 3px;

    bottom: 33px;

    right: 14px;

}}



@keyframes aaOrbit {{

    to {{

        transform: rotate(360deg);

    }}

}}



.aa-ring {{

    position: relative;

    width: 280px;

    height: 280px;

    padding: 10px;



    border-radius: 50%;



    background:

        conic-gradient(

            var(--hero-accent) {progress * 100}%,

            rgba(255,255,255,.045) 0

        );



    box-shadow:

        0 0 75px color-mix(

            in srgb,

            var(--hero-accent) 16%,

            transparent

        ),

        inset 0 0 35px rgba(0,0,0,.50);



    animation:

        aaRingEntrance 1.25s

        cubic-bezier(.16,1,.3,1) both,

        aaRingGlow 3s ease-in-out 1.3s infinite;

}}



@keyframes aaRingEntrance {{

    from {{

        opacity: 0;

        transform: scale(.68) rotate(-90deg);

    }}

    to {{

        opacity: 1;

        transform: scale(1) rotate(0);

    }}

}}



@keyframes aaRingGlow {{

    0%, 100% {{

        filter: drop-shadow(

            0 0 7px

            color-mix(in srgb, var(--hero-accent) 15%, transparent)

        );

    }}

    50% {{

        filter: drop-shadow(

            0 0 18px

            color-mix(in srgb, var(--hero-accent) 26%, transparent)

        );

    }}

}}



.aa-ring-inner {{

    width: 100%;

    height: 100%;



    border-radius: 50%;



    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;



    background:

        radial-gradient(

            circle at 48% 33%,

            rgba(34,48,61,.97),

            rgba(5,10,16,.99)

        );



    border: 1px solid rgba(255,255,255,.06);



    box-shadow:

        inset 0 0 48px rgba(0,0,0,.52),

        inset 0 1px 0 rgba(255,255,255,.025);

}}



.aa-aqi-number {{

    color: #f5f8fb;

    font-size: 82px;

    line-height: .92;

    font-weight: 850;

    letter-spacing: -.085em;

    text-shadow: 0 0 35px rgba(255,255,255,.055);

}}



.aa-aqi-label {{

    margin-top: 14px;

    color: var(--hero-accent);

    font-size: 12px;

    font-weight: 850;

    letter-spacing: .21em;

    text-transform: uppercase;

}}



.aa-subline {{

    margin-top: 10px;

    color: #7f8d9d;

    font-size: 10px;

    text-align: center;

}}



.aa-forecast-value {{

    color: #f2f6f9;

    font-size: 30px;

    line-height: 1;

    font-weight: 800;

    letter-spacing: -.045em;

}}



.aa-forecast-direction {{

    margin-top: 7px;

    color: #7f8c9c;

    font-size: 10px;

}}



.aa-bottom {{

    grid-column: 1 / -1;



    display: grid;

    grid-template-columns: repeat(3, 1fr);

    gap: 12px;

}}



.aa-stat {{

    padding: 14px 16px;

    border-radius: 15px;

    background: rgba(255,255,255,.027);

    border: 1px solid rgba(255,255,255,.055);

}}



.aa-stat-label {{

    color: #707f90;

    font-size: 8px;

    font-weight: 800;

    letter-spacing: .15em;

    text-transform: uppercase;

}}



.aa-stat-value {{

    margin-top: 5px;

    color: #dfe7ee;

    font-size: 15px;

    font-weight: 720;

}}



.aa-provider {{

    position: absolute;

    left: 31px;

    bottom: 7px;



    color: rgba(130,145,158,.52);

    font-size: 8px;

    letter-spacing: .08em;

    text-transform: uppercase;

}}



.aa-corner {{

    position: absolute;

    right: 28px;

    top: 28px;

    width: 66px;

    height: 66px;

    border-top: 1px solid rgba(255,255,255,.07);

    border-right: 1px solid rgba(255,255,255,.07);

    border-radius: 0 14px 0 0;

    opacity: .65;

}}



@media (max-width: 820px) {{

    .aa-hud {{

        grid-template-columns: 1fr;

        grid-template-rows: auto auto auto auto;

        padding: 22px;

    }}



    .aa-top,

    .aa-bottom {{

        grid-column: 1;

    }}



    .aa-side {{

        display: grid;

        grid-template-columns: 1fr 1fr;

    }}



    .aa-center {{

        min-height: 315px;

    }}



    .aa-ring {{

        width: 225px;

        height: 225px;

    }}



    .aa-aqi-number {{

        font-size: 64px;

    }}



    .aa-orbit {{

        width: 270px;

        height: 270px;

    }}



    .aa-bottom {{

        grid-template-columns: 1fr;

    }}

}}

</style>

</head>



<body>

<div class="aa-hero">



    <canvas id="aa-canvas"></canvas>

    <div class="aa-noise"></div>

    <div class="aa-sweep"></div>

    <div class="aa-corner"></div>



    <div class="aa-hud">



        <div class="aa-top">

            <div class="aa-eyebrow">

                CURRENT AIR QUALITY · LIVE MONITORING

            </div>



            <div class="aa-live-badge">

                <span class="aa-live-dot"></span>

                {safe_freshness}

            </div>

        </div>



        <div class="aa-side">



            <div class="aa-side-card">

                <div class="aa-panel-label">

                    Monitoring Station

                </div>

                <div class="aa-panel-value">

                    📍 {safe_location}

                </div>

            </div>



            <div class="aa-side-card">

                <div class="aa-panel-label">

                    Air Quality State

                </div>

                <div class="aa-panel-value">

                    {safe_category}

                </div>

            </div>



        </div>



        <div class="aa-center">



            <div class="aa-orbit"></div>



            <div class="aa-ring">



                <div class="aa-ring-inner">



                    <div class="aa-aqi-number">

                        {aqi_display}

                    </div>



                    <div class="aa-aqi-label">

                        {safe_category}

                    </div>



                    <div class="aa-subline">

                        AirAware current estimate

                    </div>



                </div>



            </div>



        </div>



        <div class="aa-side">



            <div class="aa-side-card">

                <div class="aa-panel-label">

                    Tomorrow Forecast

                </div>

                <div class="aa-forecast-value">

                    {tomorrow_text}

                </div>

            </div>



            <div class="aa-side-card">

                <div class="aa-panel-label">

                    AQI Movement

                </div>

                <div class="aa-panel-value">

                    {delta_text}

                </div>

            </div>



        </div>



        <div class="aa-bottom">



            <div class="aa-stat">

                <div class="aa-stat-label">Source</div>

                <div class="aa-stat-value">{safe_provider}</div>

            </div>



            <div class="aa-stat">

                <div class="aa-stat-label">System</div>

                <div class="aa-stat-value">AirAware ML Forecast</div>

            </div>



            <div class="aa-stat">

                <div class="aa-stat-label">Mode</div>

                <div class="aa-stat-value">Research Prototype</div>

            </div>



        </div>



    </div>

</div>



<script src="https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.min.js"></script>

<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>



<script>

(() => {{

    const hero = document.querySelector(".aa-hero");

    const canvas = document.getElementById("aa-canvas");



    // --------------------------------------------------

    // THREE.JS ATMOSPHERIC FIELD

    // --------------------------------------------------



    function startThree() {{

        if (!window.THREE) {{

            return;

        }}



        let renderer;



        try {{

            renderer = new THREE.WebGLRenderer({{

                canvas,

                antialias: true,

                alpha: true,

                powerPreference: "high-performance"

            }});

        }} catch (error) {{

            console.warn("AirAware WebGL fallback:", error);

            return;

        }}



        const width = hero.clientWidth;

        const height = hero.clientHeight;



        renderer.setPixelRatio(

            Math.min(window.devicePixelRatio || 1, 1.7)

        );



        renderer.setSize(width, height, false);



        const scene = new THREE.Scene();



        const camera = new THREE.PerspectiveCamera(

            42,

            width / height,

            .1,

            100

        );



        camera.position.z = 13;



        const count = 1500;

        const positions = new Float32Array(count * 3);

        const phases = new Float32Array(count);



        for (let i = 0; i < count; i++) {{

            const i3 = i * 3;



            positions[i3] =

                (Math.random() - .5) * 16;



            positions[i3 + 1] =

                (Math.random() - .5) * 10;



            positions[i3 + 2] =

                -Math.random() * 20;



            phases[i] =

                Math.random() * Math.PI * 2;

        }}



        const geometry =

            new THREE.BufferGeometry();



        geometry.setAttribute(

            "position",

            new THREE.BufferAttribute(

                positions,

                3

            )

        );



        const material =

            new THREE.PointsMaterial({{

                color: new THREE.Color("{accent}"),

                size: .032,

                transparent: true,

                opacity: .42,

                depthWrite: false

            }});



        const points =

            new THREE.Points(

                geometry,

                material

            );



        scene.add(points);



        const secondaryMaterial =

            material.clone();



        secondaryMaterial.color.set(

            0x5aa8ff

        );



        secondaryMaterial.opacity = .11;

        secondaryMaterial.size = .052;



        const secondary =

            new THREE.Points(

                geometry.clone(),

                secondaryMaterial

            );



        secondary.position.z = -3.5;

        scene.add(secondary);



        let mouseX = 0;

        let mouseY = 0;

        let targetX = 0;

        let targetY = 0;



        hero.addEventListener(

            "pointermove",

            (event) => {{

                const rect =

                    hero.getBoundingClientRect();



                targetX =

                    ((event.clientX - rect.left) / rect.width - .5);



                targetY =

                    ((event.clientY - rect.top) / rect.height - .5);

            }}

        );



        hero.addEventListener(

            "pointerleave",

            () => {{

                targetX = 0;

                targetY = 0;

            }}

        );



        const clock =

            new THREE.Clock();



        function render() {{

            requestAnimationFrame(render);



            const t =

                clock.getElapsedTime();



            mouseX +=

                (targetX - mouseX) * .025;



            mouseY +=

                (targetY - mouseY) * .025;



            points.rotation.y += .00022;

            secondary.rotation.y -= .00012;



            points.position.x =

                mouseX * .34;



            points.position.y =

                -mouseY * .22;



            secondary.position.x =

                -mouseX * .20;



            const array =

                geometry.attributes.position.array;



            for (let i = 0; i < count; i++) {{

                const i3 = i * 3;



                array[i3] +=

                    Math.sin(

                        t * .18 + phases[i]

                    ) * .00018;



                array[i3 + 1] +=

                    Math.cos(

                        t * .22 + phases[i]

                    ) * .00035;

            }}



            geometry.attributes.position.needsUpdate = true;



            renderer.render(

                scene,

                camera

            );

        }}



        render();



        window.addEventListener(

            "resize",

            () => {{

                const w =

                    hero.clientWidth;



                const h =

                    hero.clientHeight;



                renderer.setSize(

                    w,

                    h,

                    false

                );



                camera.aspect =

                    w / h;



                camera.updateProjectionMatrix();

            }}

        );

    }}



    // --------------------------------------------------

    // GSAP ENHANCEMENT

    // Essential content is already visible in HTML.

    // --------------------------------------------------



    function startGSAP() {{

        if (!window.gsap) {{

            return;

        }}



        gsap.fromTo(

            ".aa-top",

            {{ opacity: 0, y: -12 }},

            {{

                opacity: 1,

                y: 0,

                duration: .65,

                ease: "power3.out"

            }}

        );



        gsap.fromTo(

            ".aa-side-card",

            {{

                opacity: 0,

                y: 18

            }},

            {{

                opacity: 1,

                y: 0,

                duration: .7,

                stagger: .10,

                delay: .15,

                ease: "power3.out"

            }}

        );



        gsap.fromTo(

            ".aa-bottom .aa-stat",

            {{

                opacity: 0,

                y: 16

            }},

            {{

                opacity: 1,

                y: 0,

                duration: .55,

                stagger: .09,

                delay: .35,

                ease: "power3.out"

            }}

        );



        gsap.to(

            ".aa-orbit",

            {{

                rotate: 360,

                duration: 19,

                repeat: -1,

                ease: "none"

            }}

        );

    }}



    startThree();

    startGSAP();

}})();

</script>

</body>

</html>

"""



    components.html(

        html,

        height=585,

        scrolling=False,

    )





# ==========================================================

# LOAD MAIN DATA

# ==========================================================



df = load_historical_data()

historical_model = load_historical_model()





# ==========================================================

# SIDEBAR

# ==========================================================



st.sidebar.markdown("## 🌍 AirAware India")

st.sidebar.caption("Environmental ML research platform")

st.sidebar.divider()



page = st.sidebar.radio(

    "Navigate",

    [

        "Latest Air",

        "Historical Research",

        "Model Performance",

    ],

)



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

                locality = location.get("locality", "")



                label = (

                    f"{name} — {locality}"

                    if locality

                    else name

                )



                if label in location_options:

                    label = f"{label} (ID {location['id']})"



                location_options[label] = location



            selected_label = st.sidebar.selectbox(

                "📍 Monitoring Station",

                list(location_options.keys()),

            )



            selected_location = location_options[selected_label]



            st.sidebar.caption(

                f"{len(available_locations)} recent stations available"

            )



        else:

            st.sidebar.warning(

                "No recent monitoring stations found."

            )



    except Exception as error:

        st.sidebar.error(

            "Unable to load monitoring stations."

        )

        st.sidebar.caption(str(error))



st.sidebar.divider()

st.sidebar.caption("Research Prototype")





# ==========================================================

# GLOBAL HEADER

# ==========================================================



st.markdown(

    '<div class="aa-page-chip"><span class="dot"></span> AIR AWARE INDIA · RESEARCH PROTOTYPE</div>',

    unsafe_allow_html=True,

)



st.title("🌍 AirAware India")

st.caption(

    "Machine learning for short-term air-quality forecasting "

    "and understandable environmental insights."

)





# ==========================================================

# LATEST AIR

# ==========================================================



if page == "Latest Air":



    st.markdown(

        '<div class="aa-section-title"><h2>Latest Air</h2></div>',

        unsafe_allow_html=True,

    )

    st.markdown(

        '<div class="aa-section-subtitle">A living view of the selected monitoring station and AirAware short-term forecast.</div>',

        unsafe_allow_html=True,

    )



    if selected_location is None:

        st.error("No monitoring station is currently available.")

        st.stop()



    try:

        live = load_location_air(

            selected_location["id"],

            selected_location,

        )

    except Exception as error:

        st.error("Unable to retrieve monitoring data.")

        st.code(str(error))

        st.stop()



    location_text = live["location"]



    if live["locality"]:

        location_text += f" — {live['locality']}"



    age = live["age_hours"]



    if age > 24:

        st.warning(

            f"⚠️ Data is delayed: the latest available measurement "

            f"is {format_age(age)}. This is not presented as real-time official AQI."

        )



    pollutants = live["pollutants"]



    raw_aqi = safe_float(live["aqi"])

    live_subindices = live.get("subindices", {}) or {}



    positive_pollutants = [

        safe_float(value)

        for value in pollutants.values()

        if safe_float(value) is not None

        and safe_float(value) > 0

    ]



    positive_subindices = [

        safe_float(value)

        for value in live_subindices.values()

        if safe_float(value) is not None

        and safe_float(value) > 0

    ]



    # Prefer the calculated AQI. If an all-zero result is returned while

    # usable positive sub-indices exist, recover from those sub-indices.

    # If the station truly has no usable positive pollution observations,

    # show "Unavailable" instead of inventing an AQI of 0.

    if raw_aqi is not None and raw_aqi > 0:

        current_aqi = raw_aqi

    elif positive_subindices:

        current_aqi = max(positive_subindices)

    elif positive_pollutants:

        current_aqi = raw_aqi

    else:

        current_aqi = None



    tomorrow_prediction = None



    if current_aqi is not None:

        timestamp = live["timestamp"]



        forecast_features = pd.DataFrame(

            [

                {

                    "AQI": current_aqi,

                    "PM2.5": pollutants.get("pm25", 0),

                    "PM10": pollutants.get("pm10", 0),

                    "Day": timestamp.day,

                    "Month": timestamp.month,

                    "DayOfWeek": timestamp.weekday(),

                }

            ]

        )



        try:

            live_model = load_live_model()



            tomorrow_prediction = float(

                live_model.predict(forecast_features)[0]

            )



            tomorrow_prediction = round(

                max(0, min(tomorrow_prediction, 500))

            )

        except Exception as error:

            st.warning(f"Forecast unavailable: {error}")



    hero_category = (

        get_aqi_category(current_aqi)

        if current_aqi is not None

        else "Unavailable"

    )



    render_animated_hero(

        current_aqi=current_aqi,

        category=hero_category,

        location_text=location_text,

        freshness=live["freshness"],

        tomorrow_prediction=tomorrow_prediction,

        provider=live["provider"],

    )



    st.markdown("### Latest Pollutant Measurements")

    st.caption(

        "Measurements reported by the selected monitoring station."

    )



    pollutant_columns = st.columns(6)



    pollutant_specs = [

        ("PM2.5", "pm25", "µg/m³"),

        ("PM10", "pm10", "µg/m³"),

        ("NO₂", "no2", "µg/m³"),

        ("SO₂", "so2", "µg/m³"),

        ("CO", "co", "mg/m³"),

        ("O₃", "o3", "µg/m³"),

    ]



    for col, (label, key, unit) in zip(

        pollutant_columns,

        pollutant_specs,

    ):

        value = pollutants.get(key)



        with col:

            st.metric(

                label,

                (

                    f"{value:.2f}"

                    if value is not None

                    else "—"

                ),

                help=unit,

            )



    st.divider()



    st.markdown("### AQI Pollutant Contributions")

    st.caption(

        "Estimated CPCB-style pollutant sub-indices. "

        "The highest available sub-index determines the displayed estimated AQI."

    )



    subindices = live["subindices"]



    if subindices:

        subindex_df = pd.DataFrame(

            {

                "Pollutant": [key.upper() for key in subindices],

                "Estimated Sub-index": [

                    round(value, 1)

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

        st.info("No pollutant sub-indices available.")



    st.divider()



    st.markdown("### AirAware Insight")



    if current_aqi is not None:



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

                "Consider reducing prolonged outdoor activity if you notice discomfort."

            )

        elif current_aqi <= 300:

            insight = "Air quality is Poor."

            guidance = (

                "Pollution levels are high. Consider reducing prolonged outdoor activity, "

                "particularly if you are sensitive to air pollution."

            )

        elif current_aqi <= 400:

            insight = "Air quality is Very Poor."

            guidance = (

                "Pollution levels are very high. Consider limiting prolonged outdoor exposure "

                "and checking air-quality conditions before extended outdoor activity."

            )

        else:

            insight = "Air quality is Severe."

            guidance = (

                "Pollution levels are extremely high. Consider avoiding prolonged outdoor "

                "exposure and closely monitoring local air-quality conditions."

            )



        st.markdown(

            f"""

            <div class="aa-insight">

                <strong>{insight}</strong><br><br>

                {guidance}

            </div>

            """,

            unsafe_allow_html=True,

        )



        if tomorrow_prediction is not None:

            change = tomorrow_prediction - current_aqi

            tomorrow_category = get_aqi_category(tomorrow_prediction)



            if change < -5:

                outlook = (

                    f"AirAware forecasts AQI to decrease from "

                    f"<strong>{current_aqi:.0f} → {tomorrow_prediction:.0f}</strong> tomorrow. "

                    f"Air quality is expected to be <strong>{tomorrow_category}</strong>."

                )

            elif change > 5:

                outlook = (

                    f"AirAware forecasts AQI to increase from "

                    f"<strong>{current_aqi:.0f} → {tomorrow_prediction:.0f}</strong> tomorrow. "

                    f"Air quality is expected to remain <strong>{tomorrow_category}</strong>."

                )

            else:

                outlook = (

                    f"AirAware forecasts AQI to remain broadly similar at around "

                    f"<strong>{tomorrow_prediction:.0f}</strong> tomorrow, with air quality "

                    f"expected to be <strong>{tomorrow_category}</strong>."

                )



            st.markdown(

                f"""

                <div class="aa-note" style="margin-top:14px;">

                    <strong>Tomorrow's outlook:</strong> {outlook}

                </div>

                """,

                unsafe_allow_html=True,

            )



    else:

        st.info(

            "Air quality insight is unavailable because a current AQI estimate "

            "could not be calculated."

        )



    st.divider()



    st.markdown("### Data & Research Note")

    st.markdown(

        """

        <div class="aa-note">

        <strong>Data source:</strong> OpenAQ monitoring data.<br><br>

        <strong>Forecast:</strong> AirAware Random Forest model trained on historical

        Indian air-quality observations.<br><br>

        <strong>Important limitation:</strong> Official CPCB AQI uses prescribed

        averaging periods and sufficient observations. AirAware's current live estimate

        is therefore labelled an <strong>estimated CPCB-style AQI</strong>.

        Current monitoring data may also be delayed.

        </div>

        """,

        unsafe_allow_html=True,

    )





# ==========================================================

# HISTORICAL RESEARCH

# ==========================================================



elif page == "Historical Research":



    st.markdown(

        '<div class="aa-section-title"><h2>Historical Research</h2></div>',

        unsafe_allow_html=True,

    )

    st.markdown(

        '<div class="aa-section-subtitle">Explore the historical Indian air-quality dataset and next-day AQI forecasting model.</div>',

        unsafe_allow_html=True,

    )



    st.markdown(

        '<div class="aa-note">Historical research mode · 2015–2020 dataset</div>',

        unsafe_allow_html=True,

    )



    col1, col2 = st.columns(2)



    with col1:

        cities = sorted(df["City"].unique())



        city = st.selectbox(

            "📍 Select city",

            cities,

        )



    city_data = df[df["City"] == city]



    with col2:

        dates = sorted(

            city_data["Date"].dt.date.unique()

        )



        date = st.selectbox(

            "📅 Select date",

            dates,

        )



    selected_rows = city_data[

        city_data["Date"].dt.date == date

    ]



    if selected_rows.empty:

        st.warning("No data available for this date.")

        st.stop()



    selected = selected_rows.iloc[0]



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



    prediction = float(

        historical_model.predict(

            pd.DataFrame([selected[features]])

        )[0]

    )



    next_date = pd.Timestamp(date) + pd.Timedelta(days=1)



    actual_row = city_data[

        city_data["Date"] == next_date

    ]



    actual_next_day = None



    if not actual_row.empty:

        actual_next_day = safe_float(

            actual_row.iloc[0]["AQI"]

        )



    prediction_error = None



    if actual_next_day is not None:

        prediction_error = abs(

            prediction - actual_next_day

        )



    st.markdown("### Air Quality Overview")



    c1, c2, c3, c4 = st.columns(4)



    with c1:

        st.metric(

            "Today's AQI",

            f"{selected['AQI']:.0f}",

        )

        st.caption(

            historical_aqi_category(selected["AQI"])

        )



    with c2:

        st.metric(

            "Predicted Tomorrow",

            f"{prediction:.0f}",

        )

        st.caption(

            historical_aqi_category(prediction)

        )



    with c3:

        st.metric(

            "AQI Change",

            f"{selected['AQI_Change']:+.0f}",

        )



    with c4:

        st.metric(

            "Actual Tomorrow",

            (

                f"{actual_next_day:.0f}"

                if actual_next_day is not None

                else "—"

            ),

        )



    st.markdown("### Forecast Evaluation")



    if actual_next_day is not None:

        st.success(

            f"AirAware predicted **{prediction:.0f} AQI**. "

            f"The actual next-day AQI was **{actual_next_day:.0f} AQI**. "

            f"Prediction error: **{prediction_error:.1f} AQI points**."

        )

    else:

        st.info(

            "Historical next-day AQI is not available for this date."

        )



    st.markdown("### Pollution Indicators")



    c1, c2, c3 = st.columns(3)



    with c1:

        st.metric("PM2.5", f"{selected['PM2.5']:.2f}")



    with c2:

        st.metric("PM10", f"{selected['PM10']:.2f}")



    with c3:

        st.metric("NO₂", f"{selected['NO2']:.2f}")



    c4, c5, c6 = st.columns(3)



    with c4:

        st.metric("CO", f"{selected['CO']:.2f}")



    with c5:

        st.metric("SO₂", f"{selected['SO2']:.2f}")



    with c6:

        st.metric("O₃", f"{selected['O3']:.2f}")



    st.markdown("### Recent AQI Trend")



    history = (

        city_data[

            city_data["Date"] <= pd.Timestamp(date)

        ]

        .tail(7)[["Date", "AQI"]]

        .set_index("Date")

    )



    st.line_chart(history)





# ==========================================================

# MODEL PERFORMANCE

# ==========================================================



elif page == "Model Performance":



    st.markdown(

        '<div class="aa-section-title"><h2>Model Performance</h2></div>',

        unsafe_allow_html=True,

    )

    st.markdown(

        '<div class="aa-section-subtitle">Evaluation of the historical next-day AQI forecasting model.</div>',

        unsafe_allow_html=True,

    )



    c1, c2, c3 = st.columns(3)



    with c1:

        st.metric("Model", "Random Forest")



    with c2:

        st.metric("Test MAE", "16.20")



    with c3:

        st.metric("Test R²", "0.883")



    st.markdown("### What the Metrics Mean")



    st.markdown(

        """

        <div class="aa-note">

        <strong>MAE — 16.20 AQI points</strong><br>

        On average, the model's prediction differs from the actual next-day AQI

        by about 16 AQI points on the held-out test set.<br><br>



        <strong>R² — 0.883</strong><br>

        The model explains a large proportion of the variation in next-day AQI

        within this test set.<br><br>



        These values describe performance on the historical test data.

        They should <strong>not</strong> be interpreted as "88.3% accuracy."

        </div>

        """,

        unsafe_allow_html=True,

    )



    st.markdown("### Model Inputs")



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

        pd.DataFrame({"Feature": feature_list}),

        use_container_width=True,

        hide_index=True,

    )



    st.markdown("### Research Limitations")



    st.warning(

        """

        AirAware is currently a research prototype.



        The historical forecasting model was trained using Indian city-level

        air-quality observations.



        The latest-monitoring pipeline uses OpenAQ station measurements.

        These measurements may be delayed and are not necessarily equivalent

        to the averaging periods used for official CPCB AQI.



        Therefore AirAware should be viewed as an experimental forecasting

        and environmental information system rather than an official

        government AQI service.

        """

    )





# ==========================================================

# FOOTER

# ==========================================================



st.divider()



st.caption(

    "AirAware India • ML + Environmental Data Research Prototype"

)
