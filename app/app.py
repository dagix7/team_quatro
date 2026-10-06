"""
Team Quatro — Ride Demand Forecast Demo App
Deliverable E: Streamlit app
Inputs:  zone (one of 12 canonical zones) + date (1–14 Nov 2025)
Outputs: 24h demand forecast curve/table, peak hour,
         estimated drivers needed (trips / 1.3), expected gross fares.

Usage:
    streamlit run app/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (all relative to project root — run from there)
# ---------------------------------------------------------------------------
ASSETS      = Path("app/assets")
MODEL_PATH  = Path("models/final_model.joblib")
TEST_PATH   = Path("data/processed/master_test.csv")
WEATHER_PATH = Path("data/processed/weather_clean.csv")
EVENTS_PATH  = Path("data/processed/events_clean.csv")

FARE_PER_TRIP = 85.0   # ETB mean gross fare per trip

CANONICAL_ZONES = [
    "Ayat", "Arat Kilo", "Bole", "CMC", "Gerji",
    "Kazanchis", "Kolfe", "Lideta", "Megenagna",
    "Merkato", "Piassa", "Sarbet",
]

# ---------------------------------------------------------------------------
# Load resources (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_test_features():
    df = pd.read_csv(TEST_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_weather():
    if WEATHER_PATH.exists():
        df = pd.read_csv(WEATHER_PATH)
        df["ts"] = pd.to_datetime(df["ts"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_events():
    if EVENTS_PATH.exists():
        return pd.read_csv(EVENTS_PATH)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
LEAKAGE_FIELDS = ["avg_fare_birr", "avg_wait_min", "active_drivers",
                  "wait_time", "completed_trips", "trips"]

def predict_zone_day(zone: str, date: pd.Timestamp, test_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Return a DataFrame with columns [hour, predicted_trips]
    for the requested zone and date.
    Weather and event features are sourced from master_test.csv
    (pre-joined at pipeline build time) — no user input required.
    """
    mask = (
        (test_df["zone"] == zone) &
        (test_df["ts"].dt.date == date.date())
    )
    day_df = test_df[mask].copy().sort_values("ts")

    if day_df.empty:
        return pd.DataFrame({"hour": range(24), "predicted_trips": [0.0] * 24})

    feature_cols = [
        c for c in day_df.columns
        if c not in ["row_id", "zone", "ts"] + LEAKAGE_FIELDS
    ]

    X = day_df[feature_cols].fillna(0)

    if model is not None:
        preds = model.predict(X)
        preds = np.clip(preds, 0, None)
    else:
        # Fallback: use same_how_mean lag as a naive forecast
        if "same_how_mean" in day_df.columns:
            preds = day_df["same_how_mean"].fillna(0).clip(lower=0).values
        else:
            preds = np.zeros(len(day_df))

    result = pd.DataFrame({
        "hour": day_df["ts"].dt.hour.values,
        "predicted_trips": preds.round(1),
    })
    return result


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ride Demand Forecast — Team Quatro",
    page_icon="🚖",
    layout="centered",
)

st.title("🚖 Addis Ababa Ride Demand Forecast")
st.caption("Hackathon Demo — Nov 2025 forecast, Team Quatro")

# --- Inputs: zone + date ONLY ---
col1, col2 = st.columns(2)
with col1:
    zone = st.selectbox("Select Zone", CANONICAL_ZONES, index=CANONICAL_ZONES.index("Bole"))
with col2:
    date_input = st.date_input(
        "Select Date (1–14 Nov 2025)",
        value=pd.Timestamp("2025-11-01"),
        min_value=pd.Timestamp("2025-11-01"),
        max_value=pd.Timestamp("2025-11-14"),
    )

st.info(
    "ℹ️ Weather conditions and event schedules are automatically "
    "sourced from bundled datasets — no manual input required.",
    icon="ℹ️",
)

run_btn = st.button("Generate 24h Forecast", type="primary")

if run_btn:
    with st.spinner("Running forecast…"):
        test_df = load_test_features()
        model   = load_model()
        forecast = predict_zone_day(zone, pd.Timestamp(date_input), test_df, model)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    peak_hour      = int(forecast.loc[forecast["predicted_trips"].idxmax(), "hour"])
    peak_trips     = float(forecast["predicted_trips"].max())
    total_trips    = float(forecast["predicted_trips"].sum())
    drivers_needed = forecast["predicted_trips"].apply(lambda t: max(1, round(t / 1.3)))
    gross_fares    = round(total_trips * FARE_PER_TRIP, 2)

    # ------------------------------------------------------------------
    # Display forecast chart
    # ------------------------------------------------------------------
    st.subheader(f"📊 24h Demand Forecast — {zone} — {date_input}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(forecast["hour"], forecast["predicted_trips"], color="#2E86AB", alpha=0.85)
    ax.axvline(peak_hour, color="#E63946", linestyle="--", linewidth=2,
               label=f"Peak hour: {peak_hour}:00")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Trips")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ------------------------------------------------------------------
    # Key metrics row
    # ------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Trips (24h)",    f"{total_trips:,.0f}")
    m2.metric("Peak Hour",             f"{peak_hour}:00 ({peak_trips:.0f} trips)")
    m3.metric("Peak Drivers Needed",   f"{int(peak_trips / 1.3)}")
    m4.metric("Est. Gross Fares",      f"ETB {gross_fares:,.0f}")

    # ------------------------------------------------------------------
    # Detailed hourly table
    # ------------------------------------------------------------------
    st.subheader("Hourly Breakdown")
    table_df = forecast.copy()
    table_df["drivers_needed"]  = drivers_needed.values
    table_df["est_gross_fare_ETB"] = (table_df["predicted_trips"] * FARE_PER_TRIP).round(0)
    table_df.columns = ["Hour", "Predicted Trips", "Drivers Needed", "Est. Gross Fare (ETB)"]
    st.dataframe(table_df.set_index("Hour"), use_container_width=True)

    st.caption(
        f"**Model:** {'Trained model loaded' if model else 'Naive lag baseline (model not yet saved)'} | "
        f"**Driver formula:** ceil(trips / 1.3) | "
        f"**Fare assumption:** ETB {FARE_PER_TRIP:.0f}/trip"
    )
