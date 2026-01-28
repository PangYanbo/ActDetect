# src/regularity.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils_time import to_local_time_series

def compute_user_hex_stats(stays: pd.DataFrame, tz: str | None = None):
    """
    Compute user×hex regularity features used by HMM/baseline:
      - visit_days
      - visits
      - dwell_total
      - night_share / work_share
      - night_dwell / work_dwell
    """
    df = stays.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["hex_id"] = df["hex_id"].astype(str).replace({"": np.nan, "nan": np.nan})
    df = df.dropna(subset=["user_id","start_time","duration_min","hex_id"]).copy()
    df = df[df["duration_min"] > 0].copy()

    if tz is not None:
        df["start_time"] = to_local_time_series(df["start_time"], tz=tz, assume_utc_if_naive=True)

    df["date"] = df["start_time"].dt.date
    mid = df["start_time"] + pd.to_timedelta(df["duration_min"]/2, unit="m")
    mid_hour = mid.dt.hour + mid.dt.minute/60.0

    is_night = (mid_hour >= 20) | (mid_hour < 6)
    is_weekday = df["start_time"].dt.weekday < 5
    is_workhour = is_weekday & (mid_hour >= 9) & (mid_hour < 17)

    df["night_dwell"] = np.where(is_night, df["duration_min"], 0.0)
    df["work_dwell"]  = np.where(is_workhour, df["duration_min"], 0.0)

    g = df.groupby(["user_id","hex_id"], as_index=False).agg(
        visit_days=("date", lambda x: x.nunique()),
        visits=("hex_id","count"),
        dwell_total=("duration_min","sum"),
        night_dwell=("night_dwell","sum"),
        work_dwell=("work_dwell","sum"),
    )
    g["night_share"] = (g["night_dwell"]/g["dwell_total"].replace(0, np.nan)).fillna(0.0)
    g["work_share"]  = (g["work_dwell"]/g["dwell_total"].replace(0, np.nan)).fillna(0.0)
    return g

def infer_home_work_anchors(hex_stats: pd.DataFrame):
    """
    home_hex: max night_dwell
    work_hex: max work_dwell excluding home_hex
    """
    rows=[]
    for u,g in hex_stats.groupby("user_id", sort=False):
        g2 = g.sort_values("night_dwell", ascending=False)
        home_hex = g2.iloc[0]["hex_id"] if len(g2) else None
        g3 = g[g["hex_id"] != home_hex].sort_values("work_dwell", ascending=False)
        work_hex = g3.iloc[0]["hex_id"] if len(g3) else None
        rows.append((u, str(home_hex) if home_hex is not None else None,
                        str(work_hex) if work_hex is not None else None))
    return pd.DataFrame(rows, columns=["user_id","home_hex","work_hex"])

def make_hex_lookup(hex_stats: pd.DataFrame):
    return {(r.user_id, str(r.hex_id)): r for r in hex_stats.itertuples(index=False)}

def regularity_report(stays: pd.DataFrame, name="DATA"):
    """
    Returns a dict with per-user distributions used for comparison:
      - top_shares: DataFrame with top1_share/top3_share
      - unique_hex: Series
      - max_visit_days: Series
      - night_anchor_stability: Series
    """
    df = stays.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["hex_id"] = df["hex_id"].astype(str).replace({"": np.nan, "nan": np.nan})
    df = df.dropna(subset=["user_id","start_time","duration_min","hex_id"]).copy()
    df = df[df["duration_min"] > 0].copy()

    df["date"] = df["start_time"].dt.date

    # top1/top3 share by dwell
    uh = df.groupby(["user_id","hex_id"], as_index=False)["duration_min"].sum()
    ut = df.groupby("user_id", as_index=False)["duration_min"].sum().rename(columns={"duration_min":"total_dwell"})
    uh = uh.merge(ut, on="user_id", how="left")
    uh["rank"] = uh.groupby("user_id")["duration_min"].rank(method="first", ascending=False)
    top1 = uh[uh["rank"]==1][["user_id","duration_min","total_dwell"]].rename(columns={"duration_min":"top1_dwell"})
    top3 = uh[uh["rank"]<=3].groupby("user_id", as_index=False)["duration_min"].sum().rename(columns={"duration_min":"top3_dwell"})
    top = top1.merge(top3, on="user_id", how="left")
    top["top1_share"] = top["top1_dwell"]/top["total_dwell"].replace(0,np.nan)
    top["top3_share"] = top["top3_dwell"]/top["total_dwell"].replace(0,np.nan)

    unique_hex = df.groupby("user_id")["hex_id"].nunique().rename("unique_hex")
    visit_days = df.groupby(["user_id","hex_id"])["date"].nunique()
    max_visit_days = visit_days.groupby("user_id").max().rename("max_visit_days_per_hex")

    # night anchor stability
    mid = df["start_time"] + pd.to_timedelta(df["duration_min"]/2, unit="m")
    mid_hour = mid.dt.hour + mid.dt.minute/60.0
    is_night = (mid_hour >= 20) | (mid_hour < 6)
    df["night_dwell"] = np.where(is_night, df["duration_min"], 0.0)

    dn = df.groupby(["user_id","date","hex_id"], as_index=False)["night_dwell"].sum()
    dn = dn[dn["night_dwell"] > 0].sort_values(["user_id","date","night_dwell"], ascending=[True,True,False])
    top_night_hex_day = dn.drop_duplicates(["user_id","date"])[["user_id","date","hex_id"]]

    def _stab(g):
        if len(g)==0:
            return np.nan
        mode_hex = g["hex_id"].value_counts().idxmax()
        return (g["hex_id"]==mode_hex).mean()

    night_anchor_stability = top_night_hex_day.groupby("user_id").apply(_stab).rename("night_anchor_stability")

    print(f"\n===== {name} regularity report =====")
    print("rows:", len(df), "users:", df["user_id"].nunique())

    return {
        "top_shares": top,
        "unique_hex": unique_hex,
        "max_visit_days": max_visit_days,
        "night_anchor_stability": night_anchor_stability,
    }

def summarize_reg(name, stays_df, reg_obj):
    """
    Corrected stays/user/day: use per-user-day median, not dataset-wide unique dates.
    """
    df = stays_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df = df.dropna(subset=["user_id","start_time","duration_min"]).copy()

    calendar_days = int(df["start_time"].dt.date.nunique())
    users = int(df["user_id"].nunique())
    rows = int(len(df))

    df["date"] = df["start_time"].dt.date
    days_per_user = df.groupby("user_id")["date"].nunique()
    stays_per_user = df.groupby("user_id").size()
    spd = (stays_per_user / days_per_user).replace([np.inf,-np.inf], np.nan).dropna()

    top = reg_obj.get("top", None) or reg_obj.get("top_shares", None)
    uniq = reg_obj["unique_hex"]
    mvd = reg_obj["max_visit_days"]
    stab = pd.Series(reg_obj["night_anchor_stability"]).dropna()

    def q(s, p):
        s = pd.Series(s).dropna()
        return float(s.quantile(p)) if len(s) else np.nan

    return {
        "dataset": name,
        "users": users,
        "calendar_days": calendar_days,
        "stays": rows,
        "user_days_med": float(days_per_user.median()) if len(days_per_user) else np.nan,
        "stays/user/day_med": float(spd.median()) if len(spd) else np.nan,
        "stays/user/day_p90": float(spd.quantile(0.9)) if len(spd) else np.nan,
        "top1_share_med": q(top["top1_share"], 0.5),
        "top3_share_med": q(top["top3_share"], 0.5),
        "unique_hex_med": q(uniq, 0.5),
        "max_visit_days_med": q(mvd, 0.5),
        "night_stability_med": q(stab, 0.5),
    }