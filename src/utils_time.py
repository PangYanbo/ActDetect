# src/utils_time.py
from __future__ import annotations
import pandas as pd

def to_local_time_series(
    s: pd.Series,
    tz: str,
    assume_utc_if_naive: bool = False,
    nonexistent: str = "shift_forward",
    ambiguous: str = "infer",
) -> pd.Series:
    """
    Convert timestamps to tz-aware local time while handling DST issues.
    - If timestamps are naive:
        * assume_utc_if_naive=False: treat them as local wall time in `tz`
        * assume_utc_if_naive=True : treat them as UTC then convert to `tz`
    - Handles DST:
        nonexistent="shift_forward" prevents NonExistentTimeError (spring forward)
        ambiguous="infer" handles fall-back ambiguity when possible
    """
    s = pd.to_datetime(s, errors="coerce")
    if s.dt.tz is None:
        if assume_utc_if_naive:
            s = s.dt.tz_localize("UTC").dt.tz_convert(tz)
        else:
            s = s.dt.tz_localize(tz, nonexistent=nonexistent, ambiguous=ambiguous)
    else:
        s = s.dt.tz_convert(tz)
    return s

def week_start_monday(ts: pd.Timestamp) -> pd.Timestamp:
    """Return Monday 00:00 of the week containing ts (ts must be tz-aware or naive consistently)."""
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()

def split_cross_midnight(df: pd.DataFrame, start_col="start_time", end_col="end_time", dur_col="duration_min") -> pd.DataFrame:
    """
    Split intervals crossing midnight into two rows so each row belongs to one calendar day.
    Keeps all other columns identical.
    """
    out_rows = []
    for r in df.itertuples(index=False):
        st = getattr(r, start_col)
        en = getattr(r, end_col)
        if pd.isna(st) or pd.isna(en) or en <= st:
            continue

        if en.date() == st.date():
            out_rows.append(r)
        else:
            end_day = st.normalize() + pd.Timedelta(days=1)
            dur1 = (end_day - st).total_seconds()/60
            out_rows.append(r._replace(**{end_col: end_day, dur_col: dur1}))

            dur2 = (en - end_day).total_seconds()/60
            out_rows.append(r._replace(**{start_col: end_day, dur_col: dur2}))

    return pd.DataFrame(out_rows, columns=df.columns)