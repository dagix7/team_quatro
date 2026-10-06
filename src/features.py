"""Feature engineering for the Addis Ride Demand hackathon (Deliverables A6/A8).

Rule 6: every feature must be known at forecast time. The only history-based features
are lags of at least MIN_LAG_HOURS (14 days) and rolling levels that END MIN_LAG_HOURS
before the target hour. The test fortnight starts one hour after the last training hour,
so a 14-day lag always lands inside the training period for every test row, and
rolling-origin validation on 14-day folds is leak-free.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cleaning import ZONES

MIN_LAG_HOURS = 336            # 14 days
PRE_H, POST_H = 2, 3           # event window: 2 h before start, 3 h after end
WINDOW_TYPES = ["football_match", "concert", "conference", "exhibition", "road_closure", "sports_run"]
PHASES = ["pre", "during", "post"]
DAY_TYPES = ["public_holiday", "school_break"]
EVENT_CAP_H = 48               # cap for "hours to next / since last event"
RAIN_BINS = [-0.01, 0.1, 2.5, 7.6, np.inf]       # none / light / moderate / heavy (mm per hour)


# ---------------------------------------------------------------- stats from train
def fit_stats(train: pd.DataFrame) -> dict:
    """Everything learned from the train file only (Rule 8)."""
    t = train.dropna(subset=["trips"]).copy()
    cap = t.groupby("zone")["trips"].quantile(0.999)
    t["trips"] = t["trips"].clip(upper=t["zone"].map(cap))
    t["how"] = t["ts"].dt.dayofweek * 24 + t["ts"].dt.hour
    zone_mean = t.groupby("zone")["trips"].mean()
    how = t.groupby(["zone", "how"])["trips"].mean().unstack("how").reindex(index=ZONES, columns=range(168))
    how = how.apply(lambda r: r.fillna(zone_mean[r.name]), axis=1)    # unseen zone/hour -> zone mean
    return {"zone_cap": cap.to_dict(),
            "zone_mean": zone_mean.to_dict(),                              # fallback for level_7d / level_28d
            "zone_how_mean": {z: how.loc[z].round(4).tolist() for z in ZONES},   # fallback for the lags
            "train_start": str(train["ts"].min()), "train_end": str(train["ts"].max())}


# ------------------------------------------------------------------------ grid
def make_grid(start: str, end: str) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([ZONES, pd.date_range(start, end, freq="h")], names=["zone", "ts"])
    return idx.to_frame(index=False)


# --------------------------------------------------------------------- calendar
def add_calendar(g: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    ts = g["ts"]
    g["hour"] = ts.dt.hour
    g["dow"] = ts.dt.dayofweek
    g["is_weekend"] = (g["dow"] >= 5).astype(int)
    g["hour_of_week"] = g["dow"] * 24 + g["hour"]
    g["day_of_month"] = ts.dt.day
    g["month"] = ts.dt.month
    g["is_payday_window"] = ((g["day_of_month"] >= 25) | (g["day_of_month"] <= 3)).astype(int)

    def dates(kind):
        s = set()
        for r in events[events["event_type"] == kind].itertuples():
            s.update(pd.date_range(r.start.normalize(), r.end.normalize(), freq="D"))
        return s

    hol, brk = dates("public_holiday"), dates("school_break")
    day = ts.dt.normalize()
    g["is_holiday"] = day.isin(hol).astype(int)
    g["is_holiday_eve"] = (day + pd.Timedelta(days=1)).isin(hol).astype(int)
    g["is_school_break"] = day.isin(brk).astype(int)
    return g


# ---------------------------------------------------------------------- weather
def add_weather(g: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Many-to-one join on the EAT hour. Row count must not change."""
    w = weather.copy()
    w["rain_3h"] = w["rain_mm"].rolling(3, min_periods=1).sum()
    w["rain_6h"] = w["rain_mm"].rolling(6, min_periods=1).sum()
    w["rain_24h"] = w["rain_mm"].rolling(24, min_periods=1).sum()
    w["rain_class"] = pd.cut(w["rain_mm"], RAIN_BINS, labels=[0, 1, 2, 3]).astype(int)
    keep = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh", "rain_3h", "rain_6h", "rain_24h",
            "rain_class", "rain_missing"]
    n0 = len(g)
    out = g.merge(w[keep].reset_index(), on="ts", how="left", validate="many_to_one")
    assert len(out) == n0, "weather join changed the row count"
    return out


# ----------------------------------------------------------------------- events
def event_windows(events_long: pd.DataFrame, include_cancelled: bool = False,
                  pre_h: int = PRE_H, post_h: int = POST_H) -> pd.DataFrame:
    """One row per (event, zone, hour) with phase = pre / during / post."""
    e = events_long[~events_long["event_type"].isin(DAY_TYPES)]
    if not include_cancelled:
        e = e[e["status"] == "confirmed"]
    rows = []
    for r in e.itertuples():
        s, t = r.start.floor("h"), r.end.ceil("h")
        spans = [("pre", s - pd.Timedelta(hours=pre_h), s), ("during", s, t),
                 ("post", t, t + pd.Timedelta(hours=post_h))]
        for phase, a, b in spans:
            for h in pd.date_range(a, b, freq="h", inclusive="left"):
                rows.append((r.event_id, r.event_type, r.zone, h, phase, r.attendance))
    return pd.DataFrame(rows, columns=["event_id", "event_type", "zone", "ts", "phase", "attendance"])


def add_events(g: pd.DataFrame, events_long: pd.DataFrame, include_cancelled: bool = False) -> pd.DataFrame:
    n0 = len(g)
    win = event_windows(events_long, include_cancelled)
    flag_cols = [f"ev_{t}_{p}" for t in WINDOW_TYPES for p in PHASES]
    if len(win):
        piv = (win.assign(v=1).groupby(["zone", "ts", "event_type", "phase"])["v"].max()
                  .unstack(["event_type", "phase"]))
        piv.columns = [f"ev_{t}_{p}" for t, p in piv.columns]
        piv = piv.reindex(columns=flag_cols).reset_index()
        agg = (win.groupby(["zone", "ts"])
                  .agg(ev_n_windows=("event_id", "nunique"), _att=("attendance", "sum")).reset_index())
        agg["ev_attendance_log"] = np.log1p(agg.pop("_att"))
        g = g.merge(piv, on=["zone", "ts"], how="left", validate="one_to_one")
        g = g.merge(agg, on=["zone", "ts"], how="left", validate="one_to_one")
    else:
        for c in flag_cols + ["ev_n_windows", "ev_attendance_log"]:
            g[c] = 0
    fill = flag_cols + ["ev_n_windows", "ev_attendance_log"]
    g[fill] = g[fill].fillna(0)
    g[flag_cols + ["ev_n_windows"]] = g[flag_cols + ["ev_n_windows"]].astype(int)

    # hours until the next event starts / since the last one ended, per zone (capped)
    e = events_long[~events_long["event_type"].isin(DAY_TYPES)]
    if not include_cancelled:
        e = e[e["status"] == "confirmed"]
    g["ev_hours_to_next"] = float(EVENT_CAP_H)
    g["ev_hours_since_end"] = float(EVENT_CAP_H)
    for z, idx in g.groupby("zone").groups.items():
        ez = e[e["zone"] == z]
        if ez.empty:
            continue
        starts = np.sort(ez["start"].values.astype("datetime64[ns]"))
        ends = np.sort(ez["end"].values.astype("datetime64[ns]"))
        t = g.loc[idx, "ts"].values.astype("datetime64[ns]")
        i = np.searchsorted(starts, t, side="left")
        nxt = np.where(i < len(starts), starts[np.minimum(i, len(starts) - 1)], np.datetime64("NaT"))
        to_next = (nxt - t) / np.timedelta64(1, "h")
        j = np.searchsorted(ends, t, side="right") - 1
        prev = np.where(j >= 0, ends[np.maximum(j, 0)], np.datetime64("NaT"))
        since = (t - prev) / np.timedelta64(1, "h")
        g.loc[idx, "ev_hours_to_next"] = np.nan_to_num(np.minimum(to_next, EVENT_CAP_H), nan=EVENT_CAP_H)
        g.loc[idx, "ev_hours_since_end"] = np.nan_to_num(np.minimum(since, EVENT_CAP_H), nan=EVENT_CAP_H)
    assert len(g) == n0, "event join changed the row count"
    return g


# ------------------------------------------------------------------- lag / level
def add_lags(g: pd.DataFrame, trips: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Lag and rolling-level features. All shifted by >= MIN_LAG_HOURS (forecast-time safe)."""
    full_ts = pd.date_range(g["ts"].min(), g["ts"].max(), freq="h")
    wide = (trips.dropna(subset=["trips"]).pivot(index="ts", columns="zone", values="trips")
                 .reindex(index=full_ts, columns=ZONES))
    cap = pd.Series(stats["zone_cap"]).reindex(ZONES)
    wide = wide.clip(upper=cap, axis=1)                 # tame spikes in the history only
    base = wide.shift(MIN_LAG_HOURS)
    l336, l504, l672 = wide.shift(336), wide.shift(504), wide.shift(672)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        how = np.nanmean(np.stack([l336.values, l504.values, l672.values]), axis=0)
    parts = {
        "lag_336h": l336, "lag_504h": l504, "lag_672h": l672,
        "same_how_mean": pd.DataFrame(how, index=wide.index, columns=wide.columns),
        "level_28d": base.rolling(672, min_periods=168).mean(),
        "level_7d": base.rolling(168, min_periods=48).mean(),
    }
    long = None
    for name, df in parts.items():
        s = df.rename_axis(columns="zone").stack(future_stack=True).rename(name).reset_index()
        s = s.rename(columns={"level_0": "ts"}) if "ts" not in s.columns else s
        long = s if long is None else long.merge(s, on=["ts", "zone"], how="outer")
    long["level_ratio_7_28"] = long["level_7d"] / long["level_28d"]
    n0 = len(g)
    out = g.merge(long, on=["zone", "ts"], how="left", validate="one_to_one")
    assert len(out) == n0

    # ---- fill the holes (outage, late-launched zone, first 4 weeks of the year) - no NaN left.
    # 1) a missing lag takes the mean of the lags that exist (history only, so it is forecast-safe);
    # 2) if all three are missing, fall back to the TRAIN mean of that zone and hour-of-week;
    # 3) levels fall back to the TRAIN zone mean; a missing momentum ratio is neutral (1.0).
    lag_cols = ["lag_336h", "lag_504h", "lag_672h"]
    out["lag_n_missing"] = out[lag_cols].isna().sum(axis=1).astype(int)       # lets a model see the guess
    zi = out["zone"].map({z: i for i, z in enumerate(ZONES)}).to_numpy()
    fallback = np.array([stats["zone_how_mean"][z] for z in ZONES])[zi, out["hour_of_week"].to_numpy()]
    same_how = out["same_how_mean"].fillna(pd.Series(fallback, index=out.index))
    for c in lag_cols:
        out[c] = out[c].fillna(same_how)
    out["same_how_mean"] = same_how
    out["level_ratio_7_28"] = out["level_ratio_7_28"].fillna(1.0)
    zmean = out["zone"].map(stats["zone_mean"])
    for c in ("level_28d", "level_7d"):
        out[c] = out[c].fillna(zmean)
    return out


# --------------------------------------------------------------------- the lot
def build_features(trips_train: pd.DataFrame, test_keys: pd.DataFrame, weather: pd.DataFrame,
                   events: pd.DataFrame, events_long: pd.DataFrame, stats: dict,
                   include_cancelled: bool = False) -> pd.DataFrame:
    """Return one frame with features for every (zone, ts) in the combined grid."""
    start = min(trips_train["ts"].min(), test_keys["ts"].min()).normalize()
    end = max(trips_train["ts"].max(), test_keys["ts"].max())
    g = make_grid(str(start), str(end))
    g = add_calendar(g, events)
    g = add_weather(g, weather)
    g = add_events(g, events_long, include_cancelled)
    g = add_lags(g, trips_train, stats)
    return g


# ------------------------------------------------------------------ data dictionary
def _meta() -> list[dict]:
    m = []

    def add(name, typ, source, desc, how, known="yes"):
        m.append(dict(column=name, type=typ, source=source, description=desc, derivation=how,
                      known_at_forecast_time=known))

    add("row_id", "string", "ride_demand_train/test.csv", "Row identifier (train: record_id)", "copied; first id kept for merged duplicates")
    add("zone", "category", "all three tables", "One of 12 canonical zones", "clean_zone() label map")
    add("ts", "datetime (EAT)", "ride_demand_*.csv pickup_hour", "Start of the hour, Addis local time", "parse_mixed(); +03:00 dropped")
    add("trips", "float (target)", "ride_demand_train.csv", "Trips requested in the zone-hour (train only)",
        "negatives -> missing and dropped; x8 corrupted counts (trips/active_drivers > 2.5) divided by 8; "
        "duplicate zone-hours averaged", "no (target)")
    add("hour", "int", "ts", "Hour of day 0-23", "ts.hour")
    add("dow", "int", "ts", "Day of week, Monday=0", "ts.dayofweek")
    add("is_weekend", "int", "ts", "Saturday or Sunday", "dow >= 5")
    add("hour_of_week", "int", "ts", "0-167 position in the week", "dow*24+hour")
    add("day_of_month", "int", "ts", "Day of month", "ts.day")
    add("month", "int", "ts", "Month 1-12", "ts.month")
    add("is_payday_window", "int", "ts", "Last days / first days of the month", "day >= 25 or day <= 3")
    add("is_holiday", "int", "events_calendar.csv", "Public holiday date", "date within a public_holiday event")
    add("is_holiday_eve", "int", "events_calendar.csv", "Day before a public holiday", "next day is a holiday")
    add("is_school_break", "int", "events_calendar.csv", "Date inside a school break", "date within a school_break event")
    add("temp_c", "float", "weather_hourly.csv", "Air temperature, deg C (July F values converted)", "join on EAT hour (weather +3 h)")
    add("rain_mm", "float", "weather_hourly.csv", "Rain in the previous hour, mm (-9999 -> 0)", "join on EAT hour")
    add("humidity_pct", "float", "weather_hourly.csv", "Relative humidity %", "join on EAT hour")
    add("wind_kmh", "float", "weather_hourly.csv", "Wind speed km/h", "join on EAT hour")
    add("rain_3h", "float", "weather_hourly.csv", "Rain over the last 3 hours, mm", "rolling sum of rain_mm (3 h)")
    add("rain_6h", "float", "weather_hourly.csv", "Rain over the last 6 hours, mm", "rolling sum of rain_mm (6 h)")
    add("rain_24h", "float", "weather_hourly.csv", "Rain over the last 24 hours, mm", "rolling sum of rain_mm (24 h)")
    add("rain_class", "int 0-3", "weather_hourly.csv", "none / light / moderate / heavy", f"cut of rain_mm at {RAIN_BINS[1:-1]} mm")
    add("rain_missing", "int", "weather_hourly.csv", "Rain reading was missing and filled with 0", "sentinel / gap flag")
    for t in WINDOW_TYPES:
        for p in PHASES:
            add(f"ev_{t}_{p}", "int", "events_calendar.csv", f"{t} {p} window active in this zone",
                f"1 if hour inside the {p} window ({PRE_H} h before start / start to end / {POST_H} h after end); confirmed events only")
    add("ev_n_windows", "int", "events_calendar.csv", "Number of event windows active in the zone-hour", "count of distinct events")
    add("ev_attendance_log", "float", "events_calendar.csv", "Log of summed expected attendance of active windows",
        "log1p(sum attendance); blanks = median of type (0 for types that never state attendance)")
    add("ev_hours_to_next", "float", "events_calendar.csv", f"Hours until next event starts in the zone (cap {EVENT_CAP_H})", "searchsorted on sorted starts")
    add("ev_hours_since_end", "float", "events_calendar.csv", f"Hours since last event ended in the zone (cap {EVENT_CAP_H})", "searchsorted on sorted ends")
    add("lag_n_missing", "int 0-3", "ride_demand_train.csv", "How many of the three lags were missing and had to be filled",
        "count of NaN in lag_336h / lag_504h / lag_672h before filling")
    add("lag_336h", "float", "ride_demand_train.csv",
        "Trips in the same zone 14 days earlier (spikes capped; holes filled)",
        "shift(336 h); if missing: mean of the other lags, else train zone x hour-of-week mean")
    add("lag_504h", "float", "ride_demand_train.csv", "Trips in the same zone 21 days earlier (holes filled)",
        "shift(504 h); if missing: mean of the other lags, else train zone x hour-of-week mean")
    add("lag_672h", "float", "ride_demand_train.csv", "Trips in the same zone 28 days earlier (holes filled)",
        "shift(672 h); if missing: mean of the other lags, else train zone x hour-of-week mean")
    add("same_how_mean", "float", "ride_demand_train.csv", "Mean of the three same-hour-of-week lags",
        "nanmean(lag_336h, lag_504h, lag_672h); if all missing: train zone x hour-of-week mean")
    add("level_28d", "float", "ride_demand_train.csv", "Zone's mean trips over the 28 days ending 14 days earlier (trend level)",
        "shift(336).rolling(672); if missing (start of year / new zone): train zone mean")
    add("level_7d", "float", "ride_demand_train.csv", "Zone's mean trips over the 7 days ending 14 days earlier",
        "shift(336).rolling(168); if missing: train zone mean")
    add("level_ratio_7_28", "float", "ride_demand_train.csv", "Short vs long level (momentum)",
        "level_7d / level_28d; 1.0 when either level was missing")
    return m


FEATURE_META = _meta()
KEY_COLS = ["row_id", "zone", "ts"]
TARGET = "trips"
FEATURE_COLS = [m["column"] for m in FEATURE_META if m["column"] not in KEY_COLS + [TARGET]]