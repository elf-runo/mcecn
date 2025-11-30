# utils/visualization.py

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

def create_triage_dashboard(df: pd.DataFrame | None = None):
    """Basic placeholder for the triage dashboard."""
    st.subheader("Triage Dashboard")
    if df is None or df.empty:
        st.info("No triage data available.")
        return

    st.write("Showing basic triage summary")
    st.dataframe(df.head())

    # Example chart – adapt later
    if "triage_level" in df.columns:
        triage_counts = df["triage_level"].value_counts().reset_index()
        triage_counts.columns = ["triage_level", "count"]

        chart = (
            alt.Chart(triage_counts)
            .mark_bar()
            .encode(x="triage_level", y="count")
        )
        st.altair_chart(chart, use_container_width=True)


def create_geographic_view(df: pd.DataFrame | None = None):
    """Basic placeholder for geographic view."""
    st.subheader("Geographic View")
    if df is None or df.empty:
        st.info("No location data available.")
        return

    # Expect columns named 'lat' and 'lon'
    if not {"lat", "lon"}.issubset(df.columns):
        st.warning("No 'lat' and 'lon' columns found in data.")
        return

    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=df["lat"].mean(),
                longitude=df["lon"].mean(),
                zoom=7,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position="[lon, lat]",
                    get_radius=1000,
                    pickable=True,
                )
            ],
        )
    )


def create_trend_analysis(df: pd.DataFrame | None = None):
    """Basic placeholder for trend analysis."""
    st.subheader("Trend Analysis")
    if df is None or df.empty:
        st.info("No time-series data available.")
        return

    # Expect a datetime column like 'created_at' or 'timestamp'
    time_col = None
    for c in ["created_at", "timestamp", "time", "event_time"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        st.warning("No datetime column (e.g. 'created_at') found in data.")
        return

    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col])

    daily = tmp.groupby(tmp[time_col].dt.date).size().reset_index(name="count")

    chart = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(x=time_col, y="count")
    )
    st.altair_chart(chart, use_container_width=True)
