"""Run the whole Deliverable A pipeline:   python src/build_master.py

Reads  data/raw/*.csv   (never edited)
Writes data/processed/  master_train.csv, master_test.csv, data_dictionary_master.csv,
                        train_ops.csv (fare/wait/drivers, analysis only), *_clean.csv lookups
       reports/         A1_cleaning_log.csv, A2_*.csv, A2c_clock_proof.csv, A4_join_audit.csv,
                        A5_join_proof.txt, A6_feature_table.csv, A7_integrity_checks.txt
       models/          stats.json (everything learned from train)
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F                                     # noqa: E402
from cleaning import (EAT_OFFSET, SPIKE_TRIPS_PER_DRIVER, ZONES, CleaningLog, clean_events,  # noqa: E402
                      clean_trips, clean_weather, clean_zone, dmy_proof, parse_mixed)

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, REP, MOD = ROOT / "data/raw", ROOT / "data/processed", ROOT / "reports", ROOT / "models"
TS_FMT = "%Y-%m-%d %H:%M"

WHY = {
    "calendar": "demand follows strong hour-of-day / day-of-week rhythms and pay-period / holiday effects",
    "weather": "rain pushes riders off the street and into cars; temperature/wind shift comfort",
    "events": "crowds create spikes before, during and especially after events in the affected zone",
    "lag": "recent demand level is the best single predictor; lags >=14 days keep it forecast-safe",
}


def why(col: str) -> str:
    if col.startswith("ev_"):
        return WHY["events"]
    if col.startswith(("rain", "temp", "humid", "wind")):
        return WHY["weather"]
    if col.startswith(("lag_", "level_", "same_how")):
        return WHY["lag"]
    return WHY["calendar"]


# ------------------------------------------------------------------- clock proof
def clock_proof(weather_eat: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """A2c / B2.1: temperature peak hour and rain-demand correlation vs clock shift.

    weather_eat is the cleaned table AFTER the +3 h conversion; we undo it to get the
    file's own (UTC) clock, then test every shift from 0 to 5 hours.
    """
    w = weather_eat[weather_eat["data_type"] == "observed"]
    utc_index = w.index - EAT_OFFSET
    temp_by_hour = pd.Series(w["temp_c"].values, index=utc_index).groupby(utc_index.hour).mean()

    t = trips.dropna(subset=["trips"]).copy()
    t["how"] = t["ts"].dt.dayofweek * 24 + t["ts"].dt.hour
    t["ratio"] = t["trips"] / t.groupby(["zone", "how"])["trips"].transform("mean")
    dem = t.groupby("ts")["ratio"].mean()
    dem = dem / dem.rolling(168, min_periods=48, center=True).mean()      # remove growth trend
    rain_utc = pd.Series(w["rain_mm"].values, index=utc_index)
    rows = []
    for s in range(0, 6):
        r = rain_utc.copy()
        r.index = r.index + pd.Timedelta(hours=s)
        j = pd.concat([dem.rename("demand"), r.rename("rain")], axis=1, join="inner").dropna()
        rows.append(dict(shift_hours=s, rain_demand_corr=round(j.corr().iloc[0, 1], 4), n_hours=len(j)))
    out = pd.DataFrame(rows)
    out.attrs["temp_peak_hour_in_file"] = int(temp_by_hour.idxmax())
    out.attrs["temp_by_hour"] = temp_by_hour.round(2).to_dict()
    return out


# -------------------------------------------------------------------- validation
def validate(train, test, tmpl, weather, feats_all, n_grid_before, ops, events_long, stats) -> bool:
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print("A7 INTEGRITY CHECKS")
    check("1 one row per zone-hour in train", not train.duplicated(["zone", "ts"]).any())
    check("2 test has 4,032 unique row_ids in the original template order",
          len(test) == 4032 and test["row_id"].is_unique and test["row_id"].tolist() == tmpl["row_id"].tolist())
    check("3 only the 12 zone labels in train and test",
          set(train["zone"]) == set(ZONES) and set(test["zone"]) == set(ZONES))
    check("4 train timestamps in 2025-01-01..2025-10-31 and test in 2025-11-01..2025-11-14",
          train["ts"].min() >= pd.Timestamp("2025-01-01") and train["ts"].max() < pd.Timestamp("2025-11-01")
          and test["ts"].min() == pd.Timestamp("2025-11-01") and test["ts"].max() == pd.Timestamp("2025-11-14 23:00"))
    check("5 no negative or sentinel values left (trips, rain, temperature range)",
          (train["trips"] >= 0).all() and (weather["rain_mm"] >= 0).all() and weather["temp_c"].between(-5, 40).all())
    check("6 weather join did not change the row count", n_grid_before == len(feats_all),
          f"{n_grid_before} -> {len(feats_all)}")
    check("7 no missing values in ANY feature column of train and test (lags and levels included)",
          train[F.FEATURE_COLS].notna().all().all() and test[F.FEATURE_COLS].notna().all().all(),
          f"NaN train={int(train[F.FEATURE_COLS].isna().sum().sum())}, test={int(test[F.FEATURE_COLS].isna().sum().sum())}")
    check("8 train and test have identical feature columns",
          [c for c in train.columns if c not in ("trips",)] == list(test.columns))
    forbidden = {"avg_fare_birr", "avg_wait_min", "active_drivers", "trips"}
    check("9 no train-only column used as a feature", not (forbidden & set(F.FEATURE_COLS)))
    check("10 every history-based feature is shifted by >= 14 days", F.MIN_LAG_HOURS >= 336)
    # direct leak test: recompute lag_336h for the test rows from the training data (cap = train zone cap)
    tk = train.set_index(["zone", "ts"])["trips"]
    prev = pd.Series([tk.get((z, t - pd.Timedelta(hours=336)), np.nan) for z, t in zip(test["zone"], test["ts"])],
                     index=test.index)
    have = prev.notna() & (test["lag_n_missing"] == 0)
    capped = prev.clip(upper=test["zone"].map(stats["zone_cap"]))
    check("11 test lag_336h equals the (capped) trips of the same zone 14 days earlier",
          np.allclose(test.loc[have, "lag_336h"], capped[have]), f"{int(have.sum())} rows compared")
    check("12 test weather rows are all forecasts", (weather.loc[weather.index >= "2025-11-01", "data_type"] == "forecast").all())
    # the x8 corruption must be gone: trips per active driver back in the healthy range
    j = train.merge(ops[["row_id", "active_drivers"]], on="row_id", how="left")
    per_driver = j["trips"] / j["active_drivers"].where(j["active_drivers"] > 0)
    check("13 no x8 corrupted trip counts left (trips / active_drivers <= 2.5)",
          (per_driver.dropna() <= SPIKE_TRIPS_PER_DRIVER).all(), f"max {per_driver.max():.2f}, max trips {train['trips'].max():.0f}")
    check("14 no event has an end before its start; no road closure / holiday has a made-up crowd",
          (events_long["end"] > events_long["start"]).all()
          and (events_long.loc[events_long["event_type"].isin(["road_closure", "public_holiday", "school_break"]),
                              "attendance"] == 0).all())
    check("15 no negative wait times left in the ops table", (ops["avg_wait_min"].dropna() >= 0).all())
    return ok_all


# -------------------------------------------------------------------------- main
def main():
    for d in (PROC, REP, MOD):
        d.mkdir(parents=True, exist_ok=True)
    log = CleaningLog()

    raw_tr = pd.read_csv(RAW / "ride_demand_train.csv")
    raw_te = pd.read_csv(RAW / "ride_demand_test.csv")
    raw_w = pd.read_csv(RAW / "weather_hourly.csv")
    raw_ev = pd.read_csv(RAW / "events_calendar.csv")
    tmpl = pd.read_csv(RAW / "submission_template.csv")

    # ---- A1/A2: clean
    train = clean_trips(raw_tr, True, log, "ride_demand_train.csv")
    test = clean_trips(raw_te, False, log, "ride_demand_test.csv")
    weather = clean_weather(raw_w, log, train_end=train["ts"].max())
    events, events_long = clean_events(raw_ev, log)

    # gaps in the history (B4.2 groundwork): outage vs late launch vs random
    full = pd.date_range("2025-01-01", "2025-10-31 23:00", freq="h")
    present = train.pivot_table(index="ts", columns="zone", values="trips", aggfunc="size").reindex(full)
    all_missing = present.isna().all(axis=1)
    outage = int(all_missing.sum())
    first_seen = train.groupby("zone")["ts"].min()
    prelaunch = int(sum((full < first_seen[z]).sum() for z in ZONES))
    log.add("ride_demand_train.csv", "all", f"platform outage: {outage} consecutive hours with no rows in any zone",
            outage * len(ZONES), len(full) * len(ZONES), "left missing in the grid (never zero)", "an outage is not zero demand")
    log.add("ride_demand_train.csv", "zone", f"zone launched late (Ayat first row {first_seen['Ayat']:%Y-%m-%d})",
            prelaunch, len(full) * len(ZONES), "no rows created before launch; lags stay missing", "the zone did not exist; absent is not zero")

    # A2 label tables
    pd.concat([
        pd.DataFrame({"table": "trips_train", "raw_label": raw_tr["zone"]}).assign(clean_label=clean_zone(raw_tr["zone"])),
        pd.DataFrame({"table": "trips_test", "raw_label": raw_te["zone"]}).assign(clean_label=clean_zone(raw_te["zone"])),
        pd.DataFrame({"table": "events", "raw_label": raw_ev["zone"]}),
    ]).groupby(["table", "raw_label"], dropna=False).size().rename("n").reset_index().to_csv(REP / "A2_zone_labels_before_after.csv", index=False)
    pd.DataFrame({"raw_label": raw_ev["event_type"], "clean_label": events_label(raw_ev)}).groupby(["raw_label", "clean_label"]).size() \
        .rename("n").reset_index().to_csv(REP / "A2_event_type_labels.csv", index=False)
    time_rows = []
    tk = {"trips_train": parse_mixed(raw_tr["pickup_hour"]),
          "trips_test": parse_mixed(raw_te["pickup_hour"]),
          "weather": parse_mixed(raw_w["timestamp"]),
          "events_start": parse_mixed(raw_ev["start_datetime"]),
          "events_end": parse_mixed(raw_ev["end_datetime"])}
    raws = {"trips_train": raw_tr["pickup_hour"], "trips_test": raw_te["pickup_hour"], "weather": raw_w["timestamp"],
            "events_start": raw_ev["start_datetime"], "events_end": raw_ev["end_datetime"]}
    for k, (parsed, kinds) in tk.items():
        p = dmy_proof(raws[k], kinds)
        for kind, n in kinds.value_counts().items():
            time_rows.append(dict(table=k, format=kind, rows=int(n), unparsed=int((kinds == "unparsed").sum()),
                                  slash_rows=p["n_slash_rows"], slash_first_field_gt12=p["first_field_gt12"],
                                  slash_second_field_gt12=p["second_field_gt12"]))
    pd.DataFrame(time_rows).to_csv(REP / "A2b_timestamp_formats.csv", index=False)

    proof = clock_proof(weather, train)
    proof.to_csv(REP / "A2c_clock_proof.csv", index=False)
    pd.Series(proof.attrs["temp_by_hour"], name="mean_temp_c_by_file_hour").rename_axis("hour_in_file") \
        .to_csv(REP / "A2c_temp_by_hour.csv")

    # ---- A3/A4: features and joins
    stats = F.fit_stats(train)
    feats = F.build_features(train, test, weather, events, events_long, stats)
    n_grid = len(F.make_grid(feats["ts"].min(), feats["ts"].max()))

    keys = ["zone", "ts"]
    tr_rows = train.dropna(subset=["trips"])
    master_train = tr_rows[["row_id", "zone", "ts", "trips"]].merge(
        feats[keys + F.FEATURE_COLS], on=keys, how="left", validate="one_to_one")
    master_test = test.merge(feats[keys + F.FEATURE_COLS], on=keys, how="left", validate="one_to_one")
    assert master_test["row_id"].tolist() == test["row_id"].tolist()

    # ---- A4 audit
    raw_hours = set(parse_mixed(raw_w["timestamp"])[0] + EAT_OFFSET)
    def wx_stats(df, label):
        no_raw = ~df["ts"].isin(raw_hours)
        return dict(join=f"{label} -> weather", rows_before=len(df), rows_after=len(df),
                    match_rate_pct=round(100 * df["temp_c"].notna().mean(), 2),
                    unmatched=int(df["temp_c"].isna().sum()),
                    rows_whose_hour_had_no_raw_weather=int(no_raw.sum()),
                    note="hour missing in raw file -> interpolated (temp/humidity/wind), rain=0 + rain_missing flag")
    audit = [wx_stats(master_train, "train"), wx_stats(master_test, "test")]
    win = F.event_windows(events_long)
    keyset = set(zip(master_train["zone"], master_train["ts"])) | set(zip(master_test["zone"], master_test["ts"]))
    win["matched"] = [(z, t) in keyset for z, t in zip(win["zone"], win["ts"])]
    per_event = win.groupby("event_id")["matched"].sum()
    ev_s = events[~events["event_type"].isin(F.DAY_TYPES)]
    n_cancel = int((ev_s["status"] == "cancelled").sum())
    n_match = int((per_event > 0).sum())
    audit.append(dict(join="events (interval) -> zone-hours", rows_before=len(feats), rows_after=len(feats),
                      match_rate_pct=round(100 * n_match / max(1, (ev_s["status"] == "confirmed").sum()), 2),
                      unmatched=int((ev_s["status"] == "confirmed").sum() - n_match),
                      rows_whose_hour_had_no_raw_weather=np.nan,
                      note=f"{len(ev_s)} zone-level events: {n_match} matched >=1 zone-hour, {n_cancel} cancelled and excluded, "
                           f"{len(events) - len(ev_s)} holidays/school breaks used as day flags"))
    pd.DataFrame(audit).to_csv(REP / "A4_join_audit.csv", index=False)

    # ---- A5 join proof
    lines = ["A5 JOIN PROOF\n"]
    def show(title, row):
        r = row.iloc[0]
        lines.append(f"--- {title}: {r['zone']} at {r['ts']} (EAT), trips={r['trips']}")
        lines.append(f"weather row used (EAT {r['ts']} = file timestamp {r['ts'] - EAT_OFFSET} UTC): "
                     f"rain_mm={r['rain_mm']}, rain_3h={r['rain_3h']:.1f}, temp_c={r['temp_c']:.1f}, rain_class={r['rain_class']}")
        raw_row = raw_w[raw_w["timestamp"].str.startswith((r["ts"] - EAT_OFFSET).strftime("%Y-%m-%dT%H"))]
        lines.append("raw weather rows:\n" + raw_row.to_string(index=False))
        ev_cols = [c for c in F.FEATURE_COLS if c.startswith("ev_") and r[c] not in (0, 0.0, 48.0)]
        lines.append(f"event/holiday features set: " + ", ".join(f"{c}={r[c]}" for c in ev_cols + ["is_holiday"]))
        zev = win[(win["zone"] == r["zone"]) & (win["ts"] == r["ts"])]
        lines.append("event window rows:\n" + (zev.to_string(index=False) if len(zev) else "(none)") + "\n")
    show("rain-affected", master_train.sort_values("rain_3h", ascending=False).head(1))
    show("inside a football window", master_train[master_train["ev_football_match_during"] == 1].head(1))
    show("public holiday", master_train[master_train["is_holiday"] == 1].head(1))
    (REP / "A5_join_proof.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---- A1 / A6 / A8 exports
    log.frame().to_csv(REP / "A1_cleaning_log.csv", index=False)
    meta = pd.DataFrame(F.FEATURE_META)
    feat_tbl = meta[meta["column"].isin(F.FEATURE_COLS)].copy()
    feat_tbl["why_expected_to_help"] = feat_tbl["column"].map(why)
    feat_tbl.rename(columns={"column": "name", "derivation": "formula", "source": "source_columns"}) \
        .to_csv(REP / "A6_feature_table.csv", index=False)
    meta.to_csv(PROC / "data_dictionary_master.csv", index=False)
    assert set(meta["column"]) >= set(master_train.columns), set(master_train.columns) - set(meta["column"])

    for df in (master_train, master_test):
        df["ts"] = df["ts"].dt.strftime(TS_FMT)
    master_train.to_csv(PROC / "master_train.csv", index=False)
    master_test.to_csv(PROC / "master_test.csv", index=False)
    ops = tr_rows[["row_id", "zone", "ts", "avg_fare_birr", "avg_wait_min", "active_drivers"]].copy()
    ops["ts"] = ops["ts"].dt.strftime(TS_FMT)
    ops.to_csv(PROC / "train_ops.csv", index=False)           # analysis/demo only - NOT model inputs
    weather.reset_index().assign(ts=lambda d: d["ts"].dt.strftime(TS_FMT)).to_csv(PROC / "weather_clean.csv", index=False)
    events_long.assign(start=events_long["start"].dt.strftime(TS_FMT), end=events_long["end"].dt.strftime(TS_FMT)) \
        .to_csv(PROC / "events_clean.csv", index=False)
    (MOD / "stats.json").write_text(json.dumps(stats, indent=2))

    # ---- A7 (reloaded from disk to test what we actually exported)
    mt = pd.read_csv(PROC / "master_train.csv", parse_dates=["ts"])
    ms = pd.read_csv(PROC / "master_test.csv", parse_dates=["ts"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = validate(mt, ms, tmpl, weather, feats, n_grid, ops, events_long, stats)
        print("\nOVERALL:", "ALL PASS" if ok else "SOME CHECKS FAILED")
    text = buf.getvalue()
    (REP / "A7_integrity_checks.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"master_train {mt.shape}, master_test {ms.shape}; features: {len(F.FEATURE_COLS)}")


def events_label(raw_ev):
    from cleaning import clean_event_type
    return clean_event_type(raw_ev["event_type"])


if __name__ == "__main__":
    main()