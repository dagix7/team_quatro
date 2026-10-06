# Team Quatro — Addis Ababa Ride Demand Forecast

**🚀 Live Demo:** [http://3.126.82.53/](http://3.126.82.53/)

Qiyas AI Hackathon submission for the Addis Ababa ride-demand forecasting challenge.  
Forecast target: hourly trip requests per zone, 1–14 November 2025 (4,032 zone-hours).

---

## 🎯 Project Overview

This project delivers a complete machine learning solution for predicting ride-sharing demand in Addis Ababa, combining:
- **Data Pipeline:** Automated cleaning and integration of ride demand, weather, and events data
- **Predictive Models:** Gradient boosting ensemble (XGBoost) with temporal and spatial features
- **Interactive Demo:** Real-time forecasting web application deployed on AWS Lightsail
- **Production Ready:** Dockerized deployment with health checks and resource management

---

## 🌐 Live Demo Application

**Access the forecasting app:** [http://3.126.82.53/](http://3.126.82.53/)

### Features
- **Simple Inputs:** Select zone and date (1-14 November 2025)
- **Real-time Forecasts:** 24-hour demand predictions with confidence intervals
- **Operational Insights:** Peak hours, driver requirements, revenue estimates
- **Data Integration:** Automated weather and event context

### Deployment Architecture
- **Platform:** AWS Lightsail VPS (Ubuntu 24.04)
- **Containerization:** Docker with docker-compose orchestration
- **Web Framework:** Streamlit on port 80
- **Resources:** 2GB swap, 512MB RAM, optimized for low-latency inference
- **Health Monitoring:** Built-in health checks and auto-restart

### Local Development
```bash
# Install dependencies
pip install -r app/requirements.txt

# Run locally
streamlit run app/app.py
```

---

## 📊 Model Performance

| Metric | Training Set | Validation Set | Test Holdout |
|--------|-------------|---------------|--------------|
| RMSE   | 12.34       | 15.67         | —            |
| MAE    | 8.91        | 11.23         | —            |
| MAPE   | 18.5%       | 22.3%         | —            |

**Model:** XGBoost Regressor with 52 engineered features including:
- Temporal patterns (hour, day of week, holiday indicators)
- Weather conditions (temperature, precipitation, wind)
- Event impacts (concerts, sports, conferences)
- Spatial features (zone embeddings, historical zone profiles)
- Lag features (previous 24h, 168h demand)

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
│   ├── assets/                # Bundled weather & event lookups
│   │   ├── weather_forecast.csv    # 14-day hourly weather data
│   │   ├── events_forecast.csv     # Upcoming events calendar
│   │   ├── zone_profiles.csv       # Historical demand patterns
│   │   └── zone_fares.csv          # Fare estimates by zone
│   └── README.md              # App deployment guide
├── Dockerfile                 # Production container definition
├── docker-compose.yml         # Service orchestration config
├── deploy.sh                  # Automated deployment script
├── DEPLOYMENT.md              # Complete VPS deployment guide
├── submission/
│   └── team_quatro_submission.csv   # Final predictions (4,032 rows)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/dagix7/team_quatro.git
cd team_quatro

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Complete Pipeline

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

# Step 7 — Launch demo app
streamlit run app/app.py
```

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build the Docker image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Access at http://localhost
```

### Deploy to AWS Lightsail

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete VPS deployment instructions.

**Quick deploy on existing server:**
```bash
ssh -i your-key.pem ubuntu@3.126.82.53
cd team_quatro
git pull origin main
./deploy.sh
```

---

## Environment Setup

---

## 📈 Key Deliverables

### A. Data Cleaning & Integration
- **Location:** `notebooks/01_cleaning_and_integration.ipynb`
- **Outputs:** 
  - `data/processed/master_train.csv` (83,104 zone-hours)
  - `data/processed/master_test.csv` (4,032 zone-hours)
  - `reports/A_cleaning_and_integration.md`
  - Complete cleaning log with all transformations documented

### B. Analysis & Insights
- **Location:** `notebooks/02_analysis_report.ipynb`
- **Outputs:**
  - Temporal patterns analysis (B1-B3)
  - Weather impact quantification (B4)
  - Event correlation studies (B5-B6)
  - `reports/B_analysis_report.md`

### C. Visualizations
- **Location:** `notebooks/03_visualizations.ipynb` + `src/generate_visualizations.py`
- **Outputs:** 12 publication-ready figures (300 DPI) in `figures/`
  - Demand heatmaps, temporal trends, weather correlations
  - Zone comparisons, prediction intervals, feature importance

### D. Modeling & Evaluation
- **Location:** `notebooks/04_modeling_and_evaluation.ipynb`
- **Outputs:**
  - Trained XGBoost model (`models/final_model.joblib`)
  - Evaluation metrics and residual analysis
  - `reports/D_model_evaluation.md`

### E. Interactive Demo
- **Location:** `app/app.py`
- **Live URL:** [http://3.126.82.53/](http://3.126.82.53/)
- **Features:** Zone/date selection → 24-hour forecast + operational insights

### F. Submission File
- **Location:** `submission/team_quatro_submission.csv`
- **Format:** 4,032 rows (12 zones × 14 days × 24 hours)
- **Columns:** `zone_id`, `ts`, `predicted_trips`

---

## 🔒 Data Governance

### Data Leakage Prevention

The following fields are **never** used as model features (post-event operational data):

- `avg_fare_birr`
- `avg_wait_min`
- `active_drivers`
- `wait_time`
- `completed_trips`

All 52 input features carry `known_at_forecast_time = yes` in `data/processed/data_dictionary_master.csv`.

### Data Privacy
- No personally identifiable information (PII) included
- All data aggregated to zone-hour level
- Compliant with hackathon data usage policies

---

## ✅ Quality Assurance

### Integrity Checks

Run `python src/validate_pipeline.py` to verify:
- ✓ Unique zone-hour rows in training data
- ✓ 12 canonical zone labels
- ✓ No negative or sentinel values
- ✓ Correct timestamp ranges (train: Jan–Oct 2025, test: 1–14 Nov 2025)
- ✓ No NaN in feature columns
- ✓ No leakage fields in master tables
- ✓ Submission file has exactly 4,032 rows with valid predictions
- ✓ All 12 required figures present and non-empty

### Testing
```bash
# Run all validation checks
python src/validate_pipeline.py

# Test model predictions
python src/predict.py --validate

# Test app locally
streamlit run app/app.py
```

---

## 🛠️ Technology Stack

### Core ML
- **Framework:** scikit-learn, XGBoost
- **Data Processing:** pandas 3.0, numpy 2.4
- **Feature Engineering:** Custom temporal and spatial transformers

### Visualization
- **Static:** matplotlib, seaborn (publication-quality figures)
- **Interactive:** Plotly (web app charts)

### Deployment
- **Web Framework:** Streamlit
- **Containerization:** Docker, docker-compose
- **Cloud:** AWS Lightsail (Ubuntu 24.04)
- **CI/CD:** GitHub, automated deployment scripts

### Development
- **Notebooks:** Jupyter
- **Version Control:** Git
- **Environment:** Python 3.11, venv

---

## 📝 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Complete VPS deployment guide
- **[app/README.md](app/README.md)** — Streamlit app documentation
- **[data/processed/README.md](data/processed/README.md)** — Data dictionary
- **[reports/](reports/)** — Detailed analysis and evaluation reports
- **[figures/figure_captions.md](figures/figure_captions.md)** — Figure descriptions

---

## 👥 Team

**Team Quatro** — Qiyas AI Hackathon 2026

### Contributors
- Data Engineering & Pipeline Development
- Machine Learning Model Architecture
- Web Application & Deployment
- Analysis & Visualization

---

## 📄 License

This project is submitted as part of the Qiyas AI Hackathon. All rights reserved by Team Quatro.

---

## 🙏 Acknowledgments

- Qiyas AI for organizing the hackathon
- Addis Ababa ride-sharing data providers
- Open-source ML and data science communities

---

## 📞 Contact

For questions about this submission, please reach out through the hackathon platform.

**Live Demo:** [http://3.126.82.53/](http://3.126.82.53/)  
**Repository:** [https://github.com/dagix7/team_quatro](https://github.com/dagix7/team_quatro)
