# cleaning.py
"""Cleaning functions for the Addis Ride Demand hackathon (Deliverable A).

CLOCK CONVENTION: every table ends up on Addis Ababa local time (EAT = UTC+3, no
daylight saving), stored as naive datetimes. Evidence for the weather clock is in
build_master.py (A2c) and analysis.py (B2.1 / fig06).

Every function that fixes something writes one row to a CleaningLog, which becomes
the A1 cleaning log table.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ZONES = ["Arat Kilo", "Ayat", "Bole", "CMC", "Gerji", "Kazanchis",
         "Kolfe", "Lideta", "Megenagna", "Merkato", "Piassa", "Sarbet"]

EAT_OFFSET = pd.Timedelta(hours=3)          # weather is on UTC, trips on EAT
SPIKE_FACTOR = 8                            # corrupted trip counts are 8x the true value
SPIKE_TRIPS_PER_DRIVER = 2.5                # healthy rows are 0-1.85 trips per active driver
EVENT_TYPES = ["public_holiday", "school_break", "football_match", "concert",
               "conference", "exhibition", "road_closure", "sports_run"]

_ZONE_MAP = {
    "arat kilo": "Arat Kilo", "ayat": "Ayat", "bole": "Bole", "bole rd": "Bole",
    "c.m.c": "CMC", "cmc": "CMC", "gerji": "Gerji", "kazanchis": "Kazanchis",
    "kazanches": "Kazanchis", "kolfe": "Kolfe", "kolfe keranio": "Kolfe",
    "lideta": "Lideta", "megenagna": "Megenagna", "megenaga": "Megenagna",
    "merkato": "Merkato", "mercato": "Merkato", "piassa": "Piassa",
    "piazza": "Piassa", "sarbet": "Sarbet",
}
_CITYWIDE = {"citywide", "city-wide", "all", "all zones"}


# --------------------------------------------------------------------------- log
class CleaningLog:
    """Collects one row per issue -> A1 cleaning log."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, file, columns, issue, n, total, fix, why):
        self.rows.append(dict(
            file=file, columns=columns, issue=issue, rows_affected=int(n),
            pct_of_rows=round(100 * n / total, 2) if total else np.nan,
            fix=fix, why=why))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ------------------------------------------------------------------ labels / time
def _zone_key(x) -> str:
    x = re.sub(r"\(.*?\)", "", str(x))             # drop "(Kirkos)" style suffixes
    return re.sub(r"\s+", " ", x).strip().lower()


def clean_zone(s: pd.Series) -> pd.Series:
    """Map any raw spelling to one of the 12 canonical zones (NaN if unknown)."""
    return s.map(lambda v: _ZONE_MAP.get(_zone_key(v)))


def clean_event_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.replace(r"[\s_]+", "_", regex=True)


_RULES = [  # (name, regex, cut_to_n_chars, strptime format)
    ("iso_z", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", 19, "%Y-%m-%dT%H:%M:%S"),
    ("iso_plus03", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+03:00$", 19, "%Y-%m-%dT%H:%M:%S"),
    ("iso_space", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", None, "%Y-%m-%d %H:%M"),
    ("dmy_slash", r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$", None, "%d/%m/%Y %H:%M"),
    ("mon_ampm", r"^[A-Za-z]{3} \d{2}, \d{4} \d{2}:\d{2} [AP]M$", None, "%b %d, %Y %I:%M %p"),
]


def parse_mixed(s: pd.Series):
    """Parse every known timestamp format explicitly (never guess day/month order).

    Returns (naive datetimes exactly as written, format name per row).
    A '+03:00' offset is dropped (the wall-clock time is already EAT).
    A 'Z' suffix means UTC - the caller decides what to do with it.
    """
    s = s.astype("object").where(s.notna(), None)
    s = pd.Series([None if v is None else str(v).strip() for v in s], index=s.index, dtype="object")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    kinds = pd.Series("unparsed", index=s.index, dtype="object")
    for name, rx, cut, fmt in _RULES:
        m = s.map(lambda v: bool(v) and re.match(rx, v) is not None)
        if not m.any():
            continue
        x = s[m].str[:cut] if cut else s[m]
        out.loc[m] = pd.to_datetime(x, format=fmt).astype("datetime64[ns]")
        kinds.loc[m] = name
    return out, kinds


def dmy_proof(raw: pd.Series, kinds: pd.Series) -> dict:
    """Evidence that dd/mm/yyyy really is day-first (A2b)."""
    x = raw[kinds == "dmy_slash"].astype(str)
    a, b = x.str[:2].astype(int), x.str[3:5].astype(int)
    return dict(n_slash_rows=len(x), first_field_gt12=int((a > 12).sum()),
                second_field_gt12=int((b > 12).sum()))


# ------------------------------------------------------------------------ trips
def clean_trips(df: pd.DataFrame, is_train: bool, log: CleaningLog, name: str):
    """Clean the trip file. Train: dedupe + fix target. Test: keep rows and order."""
    d = df.copy()
    total = len(d)
    d = d.rename(columns={"record_id" if is_train else "row_id": "row_id"})

    d["zone_raw"] = d["zone"]
    d["zone"] = clean_zone(d["zone_raw"])
    assert d["zone"].notna().all(), f"unmapped zone labels in {name}: " \
        f"{d.loc[d['zone'].isna(), 'zone_raw'].unique()}"
    n_zone = int((d["zone_raw"] != d["zone"]).sum())
    log.add(name, "zone", f"inconsistent spelling/case/whitespace ({d['zone_raw'].nunique()} raw labels)",
            n_zone, total, "strip, lowercase, collapse spaces, map to 12 canonical zones",
            "the join keys and zone-level features need exactly one label per zone")

    d["ts"], kinds = parse_mixed(d["pickup_hour"])
    assert (kinds != "unparsed").all(), "unparsed timestamps in " + name
    d.attrs["time_kinds"] = kinds.value_counts().to_dict()
    d.attrs["dmy_proof"] = dmy_proof(d["pickup_hour"], kinds)
    log.add(name, "pickup_hour", "3 timestamp formats (ISO, dd/mm/yyyy, ISO with +03:00)",
            int((kinds != "iso_space").sum()), total,
            "parse each format with an explicit strptime format; drop the +03:00 suffix",
            "the +03:00 suffix means the time is already EAT, so no conversion is needed; "
            "slash dates are day-first because no second field exceeds 12")
    assert d["ts"].dt.minute.eq(0).all() and d["ts"].dt.second.eq(0).all()

    if not is_train:
        assert not d.duplicated(["zone", "ts"]).any(), "duplicate zone-hours in test file"
        return d[["row_id", "zone", "ts"]].reset_index(drop=True)

    vals = ["trips", "avg_fare_birr", "avg_wait_min", "active_drivers"]
    for c in vals:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    neg = d["trips"] < 0
    log.add(name, "trips", "negative trip counts (impossible)", neg.sum(), total,
            "set to missing, row dropped from training", "a count cannot be negative; true value unknown")
    d.loc[neg, "trips"] = np.nan
    log.add(name, "trips", "missing target", int(d["trips"].isna().sum()), total,
            "kept as missing in the hourly grid (not zero); dropped from training rows",
            "imputing a target would teach the model made-up demand")
    log.add(name, "trips", "zero-trip hours (kept, plausible at night / new zone)",
            int((d["trips"] == 0).sum()), total, "kept unchanged",
            "genuine low-demand hours; not an error")
    log.add(name, "avg_fare_birr", "missing fares", int(d["avg_fare_birr"].isna().sum()), total,
            "left missing (analysis/demo only; never a model input)", "operational column, not available at forecast time")

    # -1 is a 'no reading' sentinel in avg_wait_min (a wait cannot be negative)
    wsent = d["avg_wait_min"] < 0
    log.add(name, "avg_wait_min", "sentinel -1 for 'no reading' (impossible negative wait)", int(wsent.sum()), total,
            "set to missing (analysis only; never a model input)",
            "a -1 would drag means and correlations (B4.1) towards zero")
    d.loc[wsent, "avg_wait_min"] = np.nan

    # x8 corruption: trips for some zone-hours were multiplied by 8 while active_drivers stayed normal.
    # Evidence (train file): trips/active_drivers is ~1.3 everywhere (max 1.85 on the other rows) but
    # 126 rows sit at 2.5-14 and ~99% of them are divisible by 8; trips/8 puts them back at ~1.3.
    # Done BEFORE de-duplication so that no corrupted copy is averaged into a good one.
    per_driver = d["trips"] / d["active_drivers"].where(d["active_drivers"] > 0)
    spike = per_driver > SPIKE_TRIPS_PER_DRIVER
    log.add(name, "trips", f"x{SPIKE_FACTOR} corrupted counts (trips / active_drivers > {SPIKE_TRIPS_PER_DRIVER}; "
            f"max trips {d['trips'].max():.0f})", int(spike.sum()), total,
            f"divide by {SPIKE_FACTOR} and round to a whole trip count",
            "not real demand: they sit at 8x the zone-hour norm, mostly outside any event window, and "
            "trips/8 restores the normal trips-per-driver ratio; left in, they carry ~half of the squared error")
    d.attrs["n_spikes_fixed"] = int(spike.sum())
    d.loc[spike, "trips"] = (d.loc[spike, "trips"] / SPIKE_FACTOR).round()

    key = ["zone", "ts"]
    exact = d.duplicated(key + vals, keep="first")
    log.add(name, "all", "exact duplicate rows (same zone-hour, same values)", exact.sum(), total,
            "drop extra copies", "duplicates would double-weight those hours")
    d = d[~exact]
    conflict = int(d.duplicated(key).sum())
    log.add(name, "trips + ops columns", "duplicate zone-hours with conflicting values", conflict, total,
            "average the copies (one row per zone-hour)", "no way to tell which copy is right; the mean is the least risky")
    d = (d.groupby(key, as_index=False, sort=False)
           .agg(row_id=("row_id", "first"), trips=("trips", "mean"),
                avg_fare_birr=("avg_fare_birr", "mean"), avg_wait_min=("avg_wait_min", "mean"),
                active_drivers=("active_drivers", "mean")))
    return d.sort_values(["ts", "zone"]).reset_index(drop=True)


# ---------------------------------------------------------------------- weather
def clean_weather(w: pd.DataFrame, log: CleaningLog, train_end: pd.Timestamp,
                  name: str = "weather_hourly.csv") -> pd.DataFrame:
    """Return one row per EAT hour (complete hourly grid), no missing values."""
    d = w.copy()
    total = len(d)
    d["ts_utc"], kinds = parse_mixed(d["timestamp"])
    assert (kinds != "unparsed").all()
    d.attrs["time_kinds"] = kinds.value_counts().to_dict()
    d.attrs["dmy_proof"] = dmy_proof(d["timestamp"], kinds)
    log.add(name, "timestamp", "2 timestamp formats (ISO with Z, dd/mm/yyyy)",
            int((kinds != "iso_z").sum()), total, "explicit strptime per format",
            "slash rows are also UTC: their daily temperature peak matches the Z rows")
    d["ts"] = d["ts_utc"] + EAT_OFFSET
    log.add(name, "timestamp", "weather clock is UTC, trips are EAT (3 h behind)", total, total,
            "add 3 hours to every weather timestamp",
            "temperature peaks at 12:00 in the file (=15:00 Addis) and rain-demand correlation peaks at +3 h")

    for c in ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    sent = d["rain_mm"] < 0
    log.add(name, "rain_mm", "sentinel -9999 for 'no reading'", sent.sum(), total,
            "set to missing, then filled with 0 mm and flagged in rain_missing",
            "most hours are dry (median 0); the flag lets the model see the guess")
    d.loc[sent, "rain_mm"] = np.nan

    fh = d["temp_c"] > 40
    log.add(name, "temp_c", "values above 40 (Fahrenheit logged 10-24 July)", fh.sum(), total,
            "convert with (F-32)*5/9", "Addis never reaches 40 C; converted July values match neighbouring days")
    d.loc[fh, "temp_c"] = (d.loc[fh, "temp_c"] - 32) * 5 / 9

    d["rain_missing"] = d["rain_mm"].isna().astype(int)
    dup = int(d.duplicated("ts").sum())
    log.add(name, "all", "duplicate hours with conflicting values", dup, total,
            "average numeric columns per hour", "no basis to prefer either reading")
    d = (d.groupby("ts", as_index=True)
           .agg(temp_c=("temp_c", "mean"), rain_mm=("rain_mm", "mean"),
                humidity_pct=("humidity_pct", "mean"), wind_kmh=("wind_kmh", "mean"),
                data_type=("data_type", "first"), rain_missing=("rain_missing", "max")))

    n_h = int(d["humidity_pct"].isna().sum()); n_t = int(d["temp_c"].isna().sum())
    grid = pd.date_range(d.index.min(), d.index.max(), freq="h")
    n_gap = len(grid.difference(d.index))
    log.add(name, "timestamp", "missing hours (gaps in the hourly series)", n_gap, len(grid),
            "reindex to a full hourly grid; interpolate temp/humidity/wind (max 6 h) then carry forward/back",
            "the join must find a weather row for every zone-hour")
    d = d.reindex(grid)
    d.index.name = "ts"
    d["data_type"] = d["data_type"].ffill().bfill()
    d["rain_missing"] = d["rain_missing"].fillna(1).astype(int)

    log.add(name, "humidity_pct / temp_c", "missing readings (humidity, temperature)", n_h + n_t, len(grid),
            "linear interpolation (<=6 h); remaining gaps filled with train-period median",
            "values are smooth hour to hour; medians come from the observed (train) period only")
    for c in ["temp_c", "humidity_pct", "wind_kmh"]:
        d[c] = d[c].interpolate(limit=6, limit_direction="both")
    obs = d.loc[d.index <= train_end]
    for c in ["temp_c", "humidity_pct", "wind_kmh"]:
        d[c] = d[c].fillna(obs[c].median())
    d["humidity_pct"] = d["humidity_pct"].clip(upper=100)
    d["rain_mm"] = d["rain_mm"].fillna(0.0)
    assert d[["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]].notna().all().all()
    return d


# ----------------------------------------------------------------------- events
def clean_events(ev: pd.DataFrame, log: CleaningLog, name: str = "events_calendar.csv"):
    """Return (events, events_long): one row per event, and one row per event-zone."""
    d = ev.copy()
    total = len(d)

    raw_types = d["event_type"].nunique()
    d["event_type_raw"] = d["event_type"]
    d["event_type"] = clean_event_type(d["event_type_raw"])
    assert d["event_type"].isin(EVENT_TYPES).all(), d.loc[~d["event_type"].isin(EVENT_TYPES), "event_type"].unique()
    log.add(name, "event_type", f"inconsistent spelling ({raw_types} raw labels -> {d['event_type'].nunique()})",
            int((d["event_type_raw"] != d["event_type"]).sum()), total,
            "strip, lowercase, spaces -> underscores", "one label per event type")

    d["status_raw"] = d["status"]
    d["status"] = d["status"].astype(str).str.strip().str.lower()
    assert d["status"].isin(["confirmed", "cancelled"]).all()
    log.add(name, "status", "inconsistent case / trailing spaces", int((d["status_raw"] != d["status"]).sum()),
            total, "strip + lowercase", "cancelled events are filtered by status")

    d["start"], k1 = parse_mixed(d["start_datetime"])
    d["end"], k2 = parse_mixed(d["end_datetime"])
    assert (k1 != "unparsed").all()
    assert ((k2 != "unparsed") | d["end_datetime"].isna()).all()
    d.attrs["time_kinds"] = pd.concat([k1, k2]).value_counts().to_dict()
    d.attrs["dmy_proof"] = dmy_proof(pd.concat([d["start_datetime"], d["end_datetime"]]), pd.concat([k1, k2]))
    log.add(name, "start_datetime / end_datetime", "3 date formats incl. 'Feb 06, 2025 08:00 PM'",
            int((k1 != "iso_space").sum()), total, "explicit strptime per format (dd/mm/yyyy = day-first)",
            "avoid month/day swaps; no second field exceeds 12")

    # raw (zone strings) -> list of canonical zones
    def expand(z):
        if z is None or (isinstance(z, float) and np.isnan(z)):
            return []
        parts = [p.strip() for p in re.split(r"&|,", str(z))]
        zones: list[str] = []
        for p in parts:
            key = _zone_key(p)
            if key in _CITYWIDE:
                return list(ZONES)
            zz = _ZONE_MAP.get(key)
            assert zz is not None, f"unmapped event zone: {p!r}"
            zones.append(zz)
        return zones

    d["zones"] = d["zone"].map(expand)
    log.add(name, "zone", "spellings, '(subcity)' suffixes, '&' lists, 'Citywide/ALL'",
            int((d["zone"].astype(str).map(_zone_key).map(lambda k: k not in _ZONE_MAP)).sum()), total,
            "strip suffix, split on '&', map to canonical zones; Citywide/ALL -> all 12 zones",
            "interval join needs one (event, zone) row per affected zone")

    # duplicates: same name + type + start + zones
    # of two copies keep the one that states an attendance, then the lowest id
    # (so the original EVT-00xx row survives and the injected EVT-9xxx copy is dropped)
    d["_has_att"] = d["expected_attendance"].notna()
    d = d.sort_values(["_has_att", "event_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    dkey = d["event_name"].astype(str) + "|" + d["event_type"] + "|" + d["start"].astype(str) + "|" + d["zones"].map(lambda z: ",".join(sorted(z)))
    dup = dkey.duplicated()
    log.add(name, "all", "duplicate events (same name, type, start, zones)", dup.sum(), total,
            "keep the copy with an attendance, then the lowest event_id (not the EVT-9xxx copy)",
            "would double-count the same event")
    d = d[~dup].drop(columns="_has_att").copy()

    # end times: missing, or earlier than the start
    dur = (d["end"] - d["start"]).dt.total_seconds() / 3600
    no_end = d["end"].isna()
    inverted = ~no_end & (dur <= 0)
    bad = no_end | inverted
    med_dur = dur[~bad].groupby(d.loc[~bad, "event_type"]).median()
    d.loc[bad, "end"] = d.loc[bad].apply(
        lambda r: r["start"] + pd.Timedelta(hours=float(med_dur.get(r["event_type"], 3))), axis=1)
    for n_bad, issue in ((no_end, "missing end time"), (inverted, "end time earlier than the start")):
        log.add(name, "end_datetime", issue, int(n_bad.sum()), total,
                "end := start + median duration of that event type (learned from valid rows)",
                "windows need an end; duration per type is very regular (football 2 h, concert 4 h); "
                "for multi-day exhibitions the median duration is a guess")
    d["duration_h"] = (d["end"] - d["start"]).dt.total_seconds() / 3600

    att = d["expected_attendance"].astype(str).str.replace(",", "", regex=False).str.extract(r"(\d+)")[0].astype(float)
    n_att = int(att.isna().sum())
    type_med = att.groupby(d["event_type"]).median()               # NaN for types that never state attendance
    d["attendance_missing"] = att.isna().astype(int)
    d["attendance"] = att.fillna(d["event_type"].map(type_med)).fillna(0)
    no_data_types = sorted(type_med.index[type_med.isna()])
    log.add(name, "expected_attendance", "free text ('approx 34000'), often blank", n_att, len(d),
            "extract digits; blanks filled with the median of the same event type; flag kept; "
            f"types with no attendance at all ({', '.join(no_data_types)}) get 0, not a global median",
            "crowd size is a useful intensity feature, but a global median would invent a crowd for "
            "road closures, holidays and school breaks")

    log.add(name, "status", "events that did not go ahead (cancelled)", int((d["status"] == "cancelled").sum()), len(d),
            "excluded from event windows (config include_cancelled); footprint checked in B3.4",
            "a cancelled event cannot move riders")

    cols = ["event_id", "event_name", "event_type", "venue", "zones", "start", "end", "duration_h",
            "attendance", "attendance_missing", "status"]
    events = d[cols].reset_index(drop=True)
    long = events.explode("zones").rename(columns={"zones": "zone"})
    long = long[long["zone"].notna()].reset_index(drop=True)
    return events, long