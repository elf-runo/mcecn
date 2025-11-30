# utils/visualization.py

import pandas as pd
import plotly.express as px


def create_triage_dashboard(df: pd.DataFrame):
    """
    Build the three main figures for the Meghalaya Overview tab:
    1) Triage colour mix
    2) Cases by chief complaint
    3) Basic vital signs distribution

    Returns:
        (fig_triage, fig_complaint, fig_vitals)
    """
    # If no data, return three placeholder figs
    if df is None or df.empty:
        fig_empty = px.scatter(title="No data available")
        return fig_empty, fig_empty, fig_empty

    # 1) Triage mix
    if "triage_color" in df.columns:
        triage_counts = (
            df["triage_color"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "triage_color", "triage_color": "count"})
        )
        fig_triage = px.pie(
            triage_counts,
            names="triage_color",
            values="count",
            title="Triage Mix (RED / YELLOW / GREEN)",
        )
    else:
        fig_triage = px.scatter(title="Triage Mix – no triage_color column found")

    # 2) Complaints mix
    if "complaint" in df.columns:
        complaint_counts = (
            df["complaint"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "complaint", "complaint": "count"})
        )
        fig_complaint = px.bar(
            complaint_counts,
            x="complaint",
            y="count",
            title="Cases by Chief Complaint",
        )
    else:
        fig_complaint = px.scatter(title="Complaints – no complaint column found")

    # 3) Vital signs distribution
    vital_cols = [c for c in ["hr", "sbp", "rr", "spo2"] if c in df.columns]
    if vital_cols:
        vitals_melt = (
            df[vital_cols]
            .melt(var_name="vital_sign", value_name="value")
            .dropna(subset=["value"])
        )
        fig_vitals = px.box(
            vitals_melt,
            x="vital_sign",
            y="value",
            title="Vital Signs Distribution (Synthetic Patients)",
        )
    else:
        fig_vitals = px.scatter(title="Vitals – no hr/sbp/rr/spo2 columns found")

    return fig_triage, fig_complaint, fig_vitals


def create_geographic_view(*args, **kwargs):
    """
    Placeholder for future geographic visualisation.
    Implement when you actually call it from app.py.
    """
    return None


def create_trend_analysis(*args, **kwargs):
    """
    Placeholder for future trend analytics charts.
    Implement when you wire it into app.py.
    """
    return None
