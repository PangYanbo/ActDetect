# src/paris_qa_clean.py
from __future__ import annotations
import numpy as np
import pandas as pd

def make_dt(date_series, time_series):
    return pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str), errors="coerce")

def paris_trip_qa(trips: pd.DataFrame, hex_col: str, map_purpose):
    """
    Basic QA for Paris trips.
    Expects columns: ID, Date_O, Time_O, Date_D, Time_D, Purpose_D (+ Purpose_O optional) + hex_col
    """
    df = trips.copy()
    df["_dt_o"] = make_dt(df["Date_O"], df["Time_O"])
    df["_dt_d"] = make_dt(df["Date_D"], df["Time_D"])

    df["_ad"] = df["Purpose_D"].apply(map_purpose)
    if "Purpose_O" in df.columns:
        df["_ao"] = df["Purpose_O"].apply(map_purpose)

    df["_trip_dur_min"] = (df["_dt_d"] - df["_dt_o"]).dt.total_seconds() / 60.0

    rep = {
        "rows": int(len(df)),
        "users": int(df["ID"].nunique()),
        "missing_dt_o_rate": float(df["_dt_o"].isna().mean()),
        "missing_dt_d_rate": float(df["_dt_d"].isna().mean()),
        "missing_hex_rate": float(df[hex_col].isna().mean()) if hex_col in df.columns else np.nan,
        "missing_purposeD_map_rate": float(df["_ad"].isna().mean()),
        "neg_or_zero_trip_dur_rate": float((df["_trip_dur_min"] <= 0).mean()),
        "trip_dur_p50": float(df["_trip_dur_min"].quantile(0.5)),
        "trip_dur_p95": float(df["_trip_dur_min"].quantile(0.95)),
        "trip_dur_p99": float(df["_trip_dur_min"].quantile(0.99)),
    }

    # per-user horizon (unique origin dates)
    df["_date_o"] = df["_dt_o"].dt.date
    days_per_user = df.groupby("ID")["_date_o"].nunique()
    rep["days_per_user_median"] = float(days_per_user.median())
    rep["days_per_user_p10"] = float(days_per_user.quantile(0.10))
    rep["days_per_user_p90"] = float(days_per_user.quantile(0.90))

    return rep, df

def clean_paris_trips_light(trips_q: pd.DataFrame, hex_col: str,
                            max_trip_dur_min: float = 360.0,
                            require_hex: bool = True):
    """
    Light cleaning:
    - dt_o/d present
    - trip duration in (0, max_trip_dur_min]
    - mapped Purpose_D exists
    - (optional) hex exists
    """
    df = trips_q.copy()
    df = df[df["_dt_o"].notna() & df["_dt_d"].notna()].copy()
    df = df[(df["_trip_dur_min"] > 0) & (df["_trip_dur_min"] <= max_trip_dur_min)].copy()
    df = df[df["_ad"].notna()].copy()
    if require_hex:
        df = df[df[hex_col].notna() & (df[hex_col].astype(str).str.len() > 0)].copy()
    return df

def seq_qa_overlap_gap(trips_clean: pd.DataFrame):
    """
    Sequence QA: overlap and gap between trip-k destination and trip-(k+1) origin.
    """
    df = trips_clean.sort_values(["ID","_dt_o"]).copy()
    df["_next_dt_o"] = df.groupby("ID")["_dt_o"].shift(-1)
    df["_overlap"] = df["_next_dt_o"].notna() & (df["_next_dt_o"] < df["_dt_d"])
    df["_gap_min"] = (df["_next_dt_o"] - df["_dt_d"]).dt.total_seconds() / 60.0

    return {
        "overlap_rate": float(df["_overlap"].mean()),
        "gap_p50": float(df["_gap_min"].quantile(0.5)),
        "gap_p95": float(df["_gap_min"].quantile(0.95)),
        "gap_p99": float(df["_gap_min"].quantile(0.99)),
    }, df