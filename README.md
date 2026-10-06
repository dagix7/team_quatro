# Team Quatro — Addis Ababa Ride Demand Forecast

Hackathon submission for the Addis Ababa ride-demand forecasting challenge.
Forecast target: hourly trip requests per zone, 1–14 November 2025 (4,032 zone-hours).

---

## Repository Structure

```
team_quatro/
├── data/
│   ├── raw/                   # Original source files (read-only)
│   └── processed/             # Cleaned master tables & feature set
│       ├── master_train.csv   # 83,104 zone-hours, Jan–Oct 2025
│       ├── master_test.csv    # 4,032 zone-hours, 1–14 Nov 2025
│       └── data_dictionary_master.csv
├── notebooks/
│   ├── 01_cleaning_and_integration.ipynb
│   ├── 02_analysis_report.ipynb
│   ├── 03_visualizations.ipynb
│   └── 04_modeling_and_evaluation.ipynb
├── src/
│   ├── cleaning.py            # Data cleaning pipeline
│   ├── features.py            # Feature engineering
│   ├── train.py               # Model training
│   ├── predict.py             # Inference / submission generation
│   ├── generate_visualizations.py  # All 12 figures
│   └── validate_pipeline.py   # A7 integrity checks
├── models/
│   └── final_model.joblib     # Saved trained model
├── figures/                   # 12 PNG figures at 300 DPI
├── reports/
│   ├── A_cleaning_and_integration.md
│   ├── B_analysis_report.md
│   └── D_model_evaluation.md
├── app/
│   ├── app.py                 # Streamlit demo (Deliverable E)
│   ├── requirements.txt       # App-specific dependencies
│   └── assets/                # Bundled weather & event lookups
├── submission/
│   └── team_quatro_submission.csv   # Final predictions (4,032 rows)
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Run Order

Execute notebooks and scripts in this order from the **project root**:

```bash
# Step 1 — Data cleaning & integration
jupyter nbconvert --to notebook --execute notebooks/01_cleaning_and_integration.ipynb

# Step 2 — Exploratory analysis
jupyter nbconvert --to notebook --execute notebooks/02_analysis_report.ipynb

# Step 3 — Generate all 12 figures
python src/generate_visualizations.py

# Step 4 — Model training
python src/train.py

# Step 5 — Generate submission
python src/predict.py

# Step 6 — Run A7 integrity checks
python src/validate_pipeline.py

# Step 7 — Visualizations notebook (optional, mirrors src/generate_visualizations.py)
jupyter nbconvert --to notebook --execute notebooks/03_visualizations.ipynb
```

---

## Validation Scores

| Metric | Training Set | Validation Set |
|--------|-------------|---------------|
| RMSE   | —           | TBD after model training |
| MAE    | —           | TBD after model training |
| MAPE   | —           | TBD after model training |

> Update this table after running `python src/train.py`.

---

## Demo App Instructions

The demo app (Deliverable E) requires only **zone** and **date** inputs.
Weather and event data are automatically sourced from bundled assets.

```bash
# From the project root:
streamlit run app/app.py
```

Open the URL shown in the terminal (default: http://localhost:8501).

**Inputs:**
- Zone — select from the 12 canonical zones
- Date — any date from 1 to 14 November 2025

**Outputs:**
- 24-hour demand forecast bar chart
- Peak hour with trip count
- Estimated drivers needed (forecast trips ÷ 1.3)
- Expected gross fares in ETB (ETB 85/trip assumption)

---

## Data Leakage Policy

The following fields are **never** used as model features (post-event operational data):

- `avg_fare_birr`
- `avg_wait_min`
- `active_drivers`
- `wait_time`
- `completed_trips`

All 52 input features carry `known_at_forecast_time = yes`
in `data/processed/data_dictionary_master.csv`.

---

## Integrity Checks

Run `python src/validate_pipeline.py` to verify:
- Unique zone-hour rows in train
- 12 canonical zone labels
- No negative or sentinel values
- Correct timestamp ranges (train: Jan–Oct 2025, test: 1–14 Nov 2025)
- No NaN in feature columns
- No leakage fields in master tables
- Submission file has exactly 4,032 rows with valid predictions
- All 12 required figures present and non-empty

---

## Team

**Team Quatro** — Hackathon 2025
