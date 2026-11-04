"""
SmartFactory AI — Sensor monitoring dashboard.

Streamlit-based dashboard for real-time and historical sensor data
visualisation, fault predictions, and alert monitoring.

Run:
  streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from smartfactory_ai.config import load_config

st.set_page_config(page_title="SmartFactory AI Monitor", layout="wide")
st.title("SmartFactory AI — Sensor Monitor")

cfg = load_config(ROOT / "config" / "config.yaml")
readings_path = ROOT / cfg.storage.live_readings_csv
training_path = ROOT / "data" / "training_data.csv"
alerts_path = ROOT / cfg.storage.live_alerts_csv

source = st.sidebar.radio("Data source", ["live", "training"], index=1)
st.sidebar.button("Refresh")
path = readings_path if source == "live" else training_path

if not path.exists():
    st.warning(f"No data file found at `{path}`. Generate training data or start the collector first.")
    st.stop()


@st.cache_data(ttl=2)
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


try:
    df = load_data(str(path))
except Exception as exc:
    st.error(f"Could not read data: {exc}")
    st.stop()

if df.empty:
    st.warning("Data file is empty.")
    st.stop()

# Ensure numeric columns
for col in ["temperature", "vibration", "current", "rpm"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

latest = df.iloc[-1]

# Top-level metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Machine", str(latest.get("machine_code", "N/A")))
c2.metric("Temperature", f"{latest.get('temperature', 0):.1f} °C")
c3.metric("Vibration", f"{latest.get('vibration', 0):.2f} mm/s")
c4.metric("RPM", f"{latest.get('rpm', 0):.0f}")

# Trend charts
chart_df = df.tail(1200).copy()
chart_df["sample"] = range(len(chart_df))

left, right = st.columns(2)
with left:
    st.subheader("Temperature trend")
    if "temperature" in chart_df.columns:
        st.line_chart(chart_df.set_index("sample")[["temperature"]])
with right:
    st.subheader("Vibration trend")
    if "vibration" in chart_df.columns:
        st.line_chart(chart_df.set_index("sample")[["vibration"]])

left, right = st.columns(2)
with left:
    st.subheader("RPM & Current")
    cols = [c for c in ["rpm", "current"] if c in chart_df.columns]
    if cols:
        st.line_chart(chart_df.set_index("sample")[cols])
with right:
    st.subheader("Status distribution")
    if "status" in df.columns:
        counts = df["status"].fillna("unknown").value_counts().rename_axis("status").reset_index(name="count")
        st.bar_chart(counts.set_index("status"))
    else:
        st.info("No status column. Status is available when using training data or running inference.")

# Recent readings
st.subheader("Latest readings")
st.dataframe(df.tail(50), use_container_width=True)

# Alerts
st.subheader("AI Alerts")
if alerts_path.exists() and alerts_path.stat().st_size > 0:
    alerts = pd.read_csv(alerts_path)
    st.dataframe(alerts.tail(50), use_container_width=True)
else:
    st.info("No live alerts yet. Start the inference subscriber to generate predictions.")
