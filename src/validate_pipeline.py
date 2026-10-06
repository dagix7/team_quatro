"""
A7 Automated Pipeline Integrity Checks
Run from the project root: python src/validate_pipeline.py
All checks assert strict conditions and print PASS/FAIL status.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"{PASS} {label}")
    else:
        msg = f"{FAIL} {label}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(label)

# ---------------------------------------------------------------------------
# Load tables
# ---------------------------------------------------------------------------
print("Loading master tables...")
train = pd.read_csv("data/processed/master_train.csv")
test  = pd.read_csv("data/processed/master_test.csv")
train["ts"] = pd.to_datetime(train["ts"])
test["ts"]  = pd.to_datetime(test["ts"])

EXPECTED_ZONES = sorted([
    "Ayat", "Arat Kilo", "Bole", "CMC", "Gerji",
    "Kazanchis", "Kolfe", "Lideta", "Megenagna",
    "Merkato", "Piassa", "Sarbet"
])

LEAKAGE_FIELDS = ["avg_fare_birr", "avg_wait_min", "active_drivers",
                  "wait_time", "completed_trips"]

REQUIRED_FIGURES = [
    "fig01_gaps_and_missingness.png",
    "fig02_before_after_cleaning.png",
    "fig03_demand_trend_with_holidays.png",
    "fig04_hour_by_weekday_heatmap.png",
    "fig05_zone_profiles.png",
    "fig06_weather_timezone_check.png",
    "fig07_rain_effect.png",
    "fig08_event_study.png",
    "fig09_holiday_effects.png",
    "fig10_model_comparison.png",
    "fig11_forecast_vs_actual.png",
    "fig12_feature_importance.png",
]

print("\n=== A7 INTEGRITY CHECKS ===\n")

# 1. Unique zone-hours in train
train_dupes = train.duplicated(subset=["zone", "ts"]).sum()
check("1. No duplicate zone-hour rows in train", train_dupes == 0,
      f"{train_dupes} duplicates found")

# 2. Test has exactly 4,032 rows
check("2. Test set has exactly 4,032 rows", len(test) == 4032,
      f"Found {len(test)} rows")

# 3. Exactly 12 clean zone labels in train and test
train_zones = sorted(train["zone"].unique())
test_zones  = sorted(test["zone"].unique())
check("3. Train has exactly 12 canonical zone labels",
      train_zones == EXPECTED_ZONES,
      f"Got: {train_zones}")
check("4. Test has exactly 12 canonical zone labels",
      test_zones == EXPECTED_ZONES,
      f"Got: {test_zones}")

# 4. Timestamp ranges
train_min = train["ts"].min()
train_max = train["ts"].max()
check("5. Train timestamps within 2025-01-01 to 2025-10-31",
      str(train_min.date()) >= "2025-01-01" and str(train_max.date()) <= "2025-10-31",
      f"Range: {train_min} to {train_max}")

test_min = test["ts"].min()
test_max = test["ts"].max()
check("6. Test timestamps within 2025-11-01 to 2025-11-14",
      str(test_min.date()) >= "2025-11-01" and str(test_max.date()) <= "2025-11-14",
      f"Range: {test_min} to {test_max}")

# 5. No negative trips in train
neg_trips = (train["trips"] < 0).sum()
check("7. No negative trip values in train", neg_trips == 0,
      f"{neg_trips} negative rows found")

# 6. No negative weather values (rain_mm, humidity_pct, wind_kmh)
for col in ["rain_mm", "humidity_pct", "wind_kmh"]:
    if col in train.columns:
        neg_count = (train[col] < 0).sum()
        check(f"8. No negative {col} values in train", neg_count == 0,
              f"{neg_count} negative values")

# 7. No NaN in feature columns (excluding target 'trips')
feature_cols = [c for c in train.columns if c not in ["row_id", "trips", "ts", "zone"]]
train_nan = train[feature_cols].isnull().sum().sum()
check("9. No NaN values in train feature columns", train_nan == 0,
      f"{train_nan} NaN cells found")

test_feature_cols = [c for c in test.columns if c not in ["row_id", "ts", "zone"]]
test_nan = test[test_feature_cols].isnull().sum().sum()
check("10. No NaN values in test feature columns", test_nan == 0,
      f"{test_nan} NaN cells found")

# 8. Train and test share identical feature columns (excluding target)
train_feature_set = set(feature_cols)
test_feature_set  = set(test_feature_cols)
check("11. Train and test have identical feature columns",
      train_feature_set == test_feature_set,
      f"Train-only: {train_feature_set - test_feature_set} | Test-only: {test_feature_set - train_feature_set}")

# 9. No leakage fields in train or test
for lf in LEAKAGE_FIELDS:
    check(f"12. Leakage field '{lf}' absent from train", lf not in train.columns)
    check(f"13. Leakage field '{lf}' absent from test",  lf not in test.columns)

# 10. All lag features shifted >= 14 days (336 hours)
for lag_col in ["lag_336h", "lag_504h", "lag_672h"]:
    check(f"14. Lag column '{lag_col}' present in train", lag_col in train.columns)

# 11. Temperature within plausible range (5°C – 45°C for Addis Ababa)
if "temp_c" in train.columns:
    temp_min = train["temp_c"].min()
    temp_max = train["temp_c"].max()
    check("15. Temperature values in plausible range (5–45°C)",
          temp_min >= 5 and temp_max <= 45,
          f"Range: {temp_min:.1f} to {temp_max:.1f}°C")

# 12. Submission file check
sub_path = Path("submission/team_quatro_submission.csv")
if sub_path.exists():
    sub = pd.read_csv(sub_path)
    check("16. Submission file has exactly 4,032 rows", len(sub) == 4032,
          f"Found {len(sub)} rows")
    check("17. Submission has required columns [row_id, predicted_trips]",
          list(sub.columns) == ["row_id", "predicted_trips"])
    if "predicted_trips" in sub.columns and len(sub) > 0:
        has_nan = sub["predicted_trips"].isnull().any()
        has_neg = (sub["predicted_trips"] < 0).any()
        check("18. No NaN predicted_trips in submission", not has_nan)
        check("19. No negative predicted_trips in submission", not has_neg)
else:
    check("16. Submission file exists", False, "submission/team_quatro_submission.csv not found")

# 13. All required figures present and non-empty
for fig in REQUIRED_FIGURES:
    fig_path = Path("figures") / fig
    exists = fig_path.exists()
    non_empty = fig_path.stat().st_size > 1000 if exists else False
    check(f"20. Figure '{fig}' exists and non-empty", exists and non_empty,
          "Missing or empty" if not exists else "File too small")

# 14. figure_captions.md covers all 12 figures
captions_path = Path("figures/figure_captions.md")
if captions_path.exists():
    captions_text = captions_path.read_text(encoding="utf-8")
    all_covered = all(fig in captions_text for fig in REQUIRED_FIGURES)
    check("21. figure_captions.md covers all 12 required figures", all_covered)
else:
    check("21. figure_captions.md exists", False)

# 15. No absolute paths in source files
abs_path_hits = []
# Search for common absolute path patterns (check other files, not this script itself)
for src_file in Path("src").glob("*.py"):
    if src_file.name == "validate_pipeline.py":
        continue  # skip self — contains the pattern strings for detection
    text = src_file.read_text(encoding="utf-8")
    if "C:\\" in text or "/content/" in text or "/home/" in text:
        abs_path_hits.append(src_file.name)
check("22. No absolute paths in src/*.py files",
      len(abs_path_hits) == 0,
      f"Found in: {abs_path_hits}")

nb_abs_hits = []
for nb_file in Path("notebooks").glob("*.ipynb"):
    text = nb_file.read_text(encoding="utf-8")
    if "C:\\\\" in text or "/content/" in text:
        nb_abs_hits.append(nb_file.name)
check("23. No absolute paths in notebooks/*.ipynb",
      len(nb_abs_hits) == 0,
      f"Found in: {nb_abs_hits}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
if errors:
    print(f"OVERALL: {len(errors)} FAILURE(S)")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print("OVERALL: ALL CHECKS PASSED ✓")
