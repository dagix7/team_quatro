# generate_slides_pdf.py
"""Generate Deliverable F: 5-Slide Presentation Deck in PDF format.

Fulfills all requirements from Hackathon Section 6.1 & Deliverable F:
- 5 slides for 5-minute presentation + 2 min Q&A
- Slide 1: The Problem & The Three Tables
- Slide 2: Data Cleaning, Clock Proof & Integration (with A4 numbers and 0 NaNs)
- Slide 3: Exploratory Data Findings (with 2 embedded figures: Fig 04 Heatmap, Fig 07 Rain Effect)
- Slide 4: Modeling, Ablation & Rolling-Origin Evaluation (with Fig 10 Model Comparison, Fig 12 Importance)
- Slide 5: Error Analysis, Operational Deployment & Live Demo (with Fig 11 Forecast vs Actual)
- High-resolution 16:9 widescreen layout (1920x1080 equivalent at 150 DPI)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
OUT_PDF = REPO_ROOT / "presentation" / "team_quatro_slides.pdf"

# 16:9 Widescreen dimensions (inches)
FIG_WIDTH = 13.333
FIG_HEIGHT = 7.5
DPI = 150

# Corporate Color Palette
BG_COLOR = "#f8fafc"       # Slate 50
NAVY_HEADER = "#0f172a"    # Slate 900
COBALT = "#1e40af"         # Blue 800
ACCENT_GREEN = "#059669"   # Emerald 600
ACCENT_ORANGE = "#d97706"  # Amber 600
CARD_BG = "#ffffff"        # White
BORDER_COLOR = "#e2e8f0"   # Slate 200
TEXT_DARK = "#1e293b"      # Slate 800
TEXT_MUTED = "#64748b"     # Slate 500


def create_base_slide(slide_num: int, title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # Top Header Banner
    header_rect = Rectangle((0, 0.88), 1, 0.12, facecolor=NAVY_HEADER, edgecolor="none", zorder=1)
    ax.add_patch(header_rect)

    # Accent bar under banner
    accent_bar = Rectangle((0, 0.875), 1, 0.005, facecolor=COBALT, edgecolor="none", zorder=2)
    ax.add_patch(accent_bar)

    # Header Text
    ax.text(0.04, 0.945, f"SLIDE {slide_num} OF 5  |  QIYAS AI HACKATHON — TEAM QUATRO",
            fontsize=9, fontweight="bold", color="#94a3b8", va="center", ha="left", zorder=3)
    ax.text(0.04, 0.908, title,
            fontsize=17, fontweight="bold", color="#ffffff", va="center", ha="left", zorder=3)
    ax.text(0.96, 0.908, subtitle,
            fontsize=11, fontweight="medium", color="#cbd5e1", va="center", ha="right", zorder=3)

    # Footer
    ax.text(0.04, 0.025, "Addis Ababa Ride Demand Forecasting  •  Nov 1–14 2025 Forecast Fortnight  •  Final Submission",
            fontsize=8, color=TEXT_MUTED, va="center", ha="left")
    ax.text(0.96, 0.025, f"Team Quatro  •  Slide {slide_num}/5",
            fontsize=8, fontweight="bold", color=COBALT, va="center", ha="right")

    return fig, ax


def draw_card(ax: plt.Axes, x: float, y: float, w: float, h: float,
              title: str = "", title_color: str = COBALT, bg: str = CARD_BG):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                          facecolor=bg, edgecolor=BORDER_COLOR,
                          linewidth=1.2, zorder=2)
    ax.add_patch(card)
    if title:
        # Title bar pill
        ax.text(x + 0.02, y + h - 0.035, title,
                fontsize=11.5, fontweight="bold", color=title_color, va="center", ha="left", zorder=3)
        div = plt.Line2D([x + 0.02, x + w - 0.02], [y + h - 0.055, y + h - 0.055],
                         color=BORDER_COLOR, linewidth=0.8, zorder=3)
        ax.add_artist(div)


def build_slide_1(pdf: PdfPages):
    fig, ax = create_base_slide(1, "The Problem & The Three Tables", "Macro Context & Table Overview")

    # Left Column: Problem Framing & Target Separation
    draw_card(ax, 0.04, 0.07, 0.44, 0.77, "The Operational Forecasting Challenge")
    ax.text(0.06, 0.77, "Addis Ababa Ride-Hailing Platform Dispatch", fontsize=12, fontweight="bold", color=TEXT_DARK)
    
    body_text_1 = (
        "• Goal: Forecast hourly pickup demand across exactly 12 canonical zones in\n"
        "  Addis Ababa for the 14-day test fortnight (November 1–14, 2025: 4,032 zone-hours).\n\n"
        "• Core Objective: Enable operations managers to pre-position fleet capacity,\n"
        "  minimize passenger wait times, and prevent localized supply shortages.\n\n"
        "• The Zero-Leakage Mandate (Rule 6):\n"
        "  - Operational columns (active_drivers, avg_wait_min, avg_fare_birr) are\n"
        "    CONSEQUENCES of demand, never known at forecast time.\n"
        "  - Pipeline strictly drops all 3 operational features prior to training to eliminate\n"
        "    artificial 20% score inflation and guarantee real-world generalization.\n\n"
        "• Horizon & Granularity:\n"
        "  - Training: Jan 1 – Oct 31, 2025 (83,104 cleaned zone-hours, 10 months).\n"
        "  - Testing: Nov 1 – Nov 14, 2025 (4,032 zone-hours, exactly 336 hours × 12 zones).\n"
        "  - Target variable: trips (count of completed ride requests per zone-hour)."
    )
    ax.text(0.06, 0.50, body_text_1, fontsize=9.5, color=TEXT_DARK, va="center", linespacing=1.35)

    # Right Column: The Three Tables Architecture
    draw_card(ax, 0.52, 0.07, 0.44, 0.77, "The Three Raw Export Tables")
    
    tables_info = [
        ("1. Trip Demand History (ride_demand_train.csv & test.csv)",
         "• 85,460 raw records with 12 canonical zones, timestamps, and target.\n"
         "• Primary spatial-temporal backbone for hourly demand modeling.\n"
         "• Cleaned: 8 aliases unified, 24 negative targets dropped, duplicates averaged."),
        ("2. Hourly Weather Archive & Forecasts (weather_hourly.csv)",
         "• 7,538 hourly observations spanning Jan 1 – Nov 14, 2025.\n"
         "• Features: temp_c, rain_mm, humidity_pct, wind_kmh.\n"
         "• Key Issue: Timestamps on UTC clock (+3h mismatch) + Fahrenheit contamination."),
        ("3. City Events & Civic Calendar (events_calendar.csv)",
         "• 165 scheduled public events: football matches, holidays, runs, concerts.\n"
         "• Multi-zone & citywide geographic scope with expected attendance.\n"
         "• Captures localized crowd influx and holiday business shutdowns.")
    ]

    y_pos = 0.72
    for title, desc in tables_info:
        badge = Rectangle((0.54, y_pos - 0.02), 0.40, 0.035, facecolor="#eff6ff", edgecolor="#bfdbfe", zorder=3)
        ax.add_patch(badge)
        ax.text(0.55, y_pos - 0.003, title, fontsize=10, fontweight="bold", color=COBALT, zorder=4)
        ax.text(0.55, y_pos - 0.09, desc, fontsize=8.8, color=TEXT_DARK, va="top", linespacing=1.3, zorder=3)
        y_pos -= 0.22

    pdf.savefig(fig)
    plt.close(fig)


def build_slide_2(pdf: PdfPages):
    fig, ax = create_base_slide(2, "Cleaning, Clock Proof & Integration", "Deliverables A1–A8 Pipeline Architecture")

    # Left Column: Integration Metrics & Clock Proof
    draw_card(ax, 0.04, 0.07, 0.46, 0.77, "Clock Proof & Missing Weather Resolution (A4)")
    
    clock_proof_text = (
        "• The Clock Proof (UTC -> Africa/Addis_Ababa, UTC+3):\n"
        "  - Raw trip table operates on Addis local time (EAT = UTC+3).\n"
        "  - Raw weather table reached minimum at 03:00 UTC and peak at 11:00 UTC.\n"
        "  - Converting to EAT aligns solar peak at 14:00 local time (matching physics).\n\n"
        "• Resolving Missing Weather Data (Deliverables A4 & A7):\n"
        "  - Audit: 186 trip hours lacked raw weather (station dropout gaps).\n"
        "  - Unit Fix: 353 readings > 45°C converted from Fahrenheit: (F-32)*5/9.\n"
        "  - Reindexing: Deduplicated weather reindexed to continuous hourly grid\n"
        "    covering 2024-12-30 00:00 to 2025-11-15 23:00 (7,680 hours).\n"
        "  - Time-Series Interpolation: Linear time interpolation + ffill/bfill for\n"
        "    temperature, humidity, and wind. Dry baseline (0.0mm) for rainfall.\n"
        "  - Outcome: Exactly 0 NaNs across master_train.csv and master_test.csv!"
    )
    ax.text(0.06, 0.60, clock_proof_text, fontsize=9.2, color=TEXT_DARK, va="center", linespacing=1.35)

    # Mini Table of Join Audit
    audit_data = [
        ("Metric", "Weather Join", "Events Join"),
        ("Join Strategy", "Left Many-to-One (ts)", "Interval + As-Of Proximity"),
        ("Left Table Rows", "83,104", "83,104 (Invariant)"),
        ("Match Rate", "100.0% (Post-Impute)", "100.0% Coverage"),
        ("Unresolved NaNs", "0 (Zero NaNs)", "0 (Zero NaNs)"),
    ]
    ty = 0.28
    for row_idx, (c1, c2, c3) in enumerate(audit_data):
        bg = "#f1f5f9" if row_idx == 0 else ("#ffffff" if row_idx % 2 == 1 else "#f8fafc")
        fw = "bold" if row_idx == 0 or row_idx == 4 else "normal"
        color = COBALT if row_idx == 4 else TEXT_DARK
        ax.add_patch(Rectangle((0.06, ty - 0.035), 0.42, 0.04, facecolor=bg, edgecolor="#cbd5e1", zorder=3))
        ax.text(0.07, ty - 0.015, c1, fontsize=8.5, fontweight=fw, color=color, zorder=4)
        ax.text(0.24, ty - 0.015, c2, fontsize=8.5, fontweight=fw, color=color, zorder=4)
        ax.text(0.36, ty - 0.015, c3, fontsize=8.5, fontweight=fw, color=color, zorder=4)
        ty -= 0.042

    # Right Column: Visual Verification (Clock Proof Figure)
    draw_card(ax, 0.52, 0.07, 0.44, 0.77, "Visual Evidence: Timezone Alignment (Fig 06)")
    fig06_path = FIG_DIR / "fig06_weather_timezone_check.png"
    if fig06_path.exists():
        img = mpimg.imread(fig06_path)
        img_ax = fig.add_axes([0.535, 0.18, 0.41, 0.58], zorder=4)
        img_ax.imshow(img)
        img_ax.axis("off")
        ax.text(0.54, 0.12, "Diurnal Solar Proof: Peak temperature aligns at 14:00 EAT\nConfirms UTC+3 clock alignment between trip and weather series.",
                fontsize=8.5, color=TEXT_MUTED, va="center")

    pdf.savefig(fig)
    plt.close(fig)


def build_slide_3(pdf: PdfPages):
    fig, ax = create_base_slide(3, "What the Data Says: Key Empirical Findings", "Exploratory Insights from Reports B & C")

    # Left Panel: Diurnal Rhythm & Weekend Commute Inversion
    draw_card(ax, 0.04, 0.07, 0.45, 0.77, "Finding 1: Commuter Peaks vs Weekend Inversion")
    fig04_path = FIG_DIR / "fig04_hour_by_weekday_heatmap.png"
    if fig04_path.exists():
        img = mpimg.imread(fig04_path)
        img_ax = fig.add_axes([0.055, 0.32, 0.42, 0.46], zorder=4)
        img_ax.imshow(img)
        img_ax.axis("off")
    
    desc_1 = (
        "• Bimodal Weekday Peaks: Mon–Fri demand shows sharp surges at 07:00–09:00\n"
        "  (morning commute) and 17:00–20:00 (evening return, peaking at 145 trips/zone-hr).\n"
        "• Weekend Shift: Sat–Sun eliminates morning commute spikes, replaced by a smooth\n"
        "  afternoon/evening leisure plateau (13:00–22:00) concentrated in Bole & Kazanchis."
    )
    ax.text(0.06, 0.19, desc_1, fontsize=8.8, color=TEXT_DARK, va="top", linespacing=1.35)

    # Right Panel: Precipitation & Demand Elasticity
    draw_card(ax, 0.51, 0.07, 0.45, 0.77, "Finding 2: Rain Surge & Saturation Elasticity")
    fig07_path = FIG_DIR / "fig07_rain_effect.png"
    if fig07_path.exists():
        img = mpimg.imread(fig07_path)
        img_ax = fig.add_axes([0.525, 0.32, 0.42, 0.46], zorder=4)
        img_ax.imshow(img)
        img_ax.axis("off")

    desc_2 = (
        "• Immediate Rain Lift: Rain of 1–5 mm creates an immediate +18% to +35% surge\n"
        "  in ride-hailing demand as pedestrians and bus commuters seek dry transit.\n"
        "• Non-Linear Saturation: Beyond 10 mm/hr, flooding and road congestion reduce\n"
        "  driver speeds; rolling 3h rain (rain_3h) captures persistent waterlogged travel."
    )
    ax.text(0.53, 0.19, desc_2, fontsize=8.8, color=TEXT_DARK, va="top", linespacing=1.35)

    pdf.savefig(fig)
    plt.close(fig)


def build_slide_4(pdf: PdfPages):
    fig, ax = create_base_slide(4, "Modeling, Ablation & Rolling-Origin Evaluation", "Deliverable D: Method Hygiene & Validation Rigor")

    # Left Column: Model Comparison & Ablation
    draw_card(ax, 0.04, 0.07, 0.46, 0.77, "Model Comparison & Ablation Study (D1–D5)")
    fig10_path = FIG_DIR / "fig10_model_comparison.png"
    if fig10_path.exists():
        img = mpimg.imread(fig10_path)
        img_ax = fig.add_axes([0.055, 0.42, 0.43, 0.36], zorder=4)
        img_ax.imshow(img)
        img_ax.axis("off")

    ablation_text = (
        "• Baseline Benchmark: Seasonal-Naive (RMSE 17.16) beats Mean (30.16) by 2x.\n"
        "• Model Selection: LightGBM (RMSE 15.72, fit=3.2s) matches Random Forest\n"
        "  (15.78, fit=27s) at 8x training speed with native categorical zone handling.\n"
        "• Multi-Table Ablation Progression (October Holdout):\n"
        "  - Calendar + Zone + Trend : RMSE 16.92 | MAE 8.00\n"
        "  - + Weather Features      : RMSE 15.99 (Δ -0.93 trips/hr lift)\n"
        "  - + Events Features       : RMSE 16.72 (Δ -0.20 trips/hr lift)\n"
        "  - + Both (Final Model)    : RMSE 15.72 | MAE 6.78 (Δ -1.20 trips lift)"
    )
    ax.text(0.06, 0.24, ablation_text, fontsize=8.5, color=TEXT_DARK, va="center", linespacing=1.3)

    # Right Column: Rolling-Origin & Feature Importance
    draw_card(ax, 0.52, 0.07, 0.44, 0.77, "Feature Importances & Rolling Folds (D3, D6)")
    fig12_path = FIG_DIR / "fig12_feature_importance.png"
    if fig12_path.exists():
        img = mpimg.imread(fig12_path)
        img_ax = fig.add_axes([0.535, 0.36, 0.41, 0.42], zorder=4)
        img_ax.imshow(img)
        img_ax.axis("off")

    rolling_text = (
        "• Rolling-Origin Cross-Validation (4 × 14-day expanding folds):\n"
        "  - LightGBM: 13.998 ± 2.633 RMSE vs Seasonal-Naive: 16.018 ± 2.494\n"
        "  - Outperforms naive baseline in 100% of validation folds.\n"
        "• Top Predictive Signals: hour (diurnal), zone (spatial baseline),\n"
        "  days_elapsed (macro growth), temp_c & rain_3h (weather), event proximity."
    )
    ax.text(0.54, 0.18, rolling_text, fontsize=8.5, color=TEXT_DARK, va="center", linespacing=1.3)

    pdf.savefig(fig)
    plt.close(fig)


def build_slide_5(pdf: PdfPages):
    fig, ax = create_base_slide(5, "Error Analysis, Operational Deployment & Demo", "Deliverables D7–D9, E & Real-World Operations")

    # Left Column: Error Analysis & Actual vs Forecast
    draw_card(ax, 0.04, 0.07, 0.46, 0.77, "Error Distribution & Forecast Fit (D7 & Fig 11)")
    fig11_path = FIG_DIR / "fig11_forecast_vs_actual.png"
    if fig11_path.exists():
        img = mpimg.imread(fig11_path)
        img_ax = fig.add_axes([0.055, 0.40, 0.43, 0.38], zorder=4)
        img_ax.imshow(img)
        img_ax.axis("off")

    error_text = (
        "• Error Concentration:\n"
        "  - Largest errors occur during morning rush (07:00) and evening peak (18:00)\n"
        "    in high-volume commercial zones (CMC, Merkato, Kazanchis).\n"
        "  - Extreme outlier spikes (>400 trips) correspond to unscheduled civic surges\n"
        "    outside the official calendar. Baseline zones (Ayat, Arat Kilo) have MAE < 4.8.\n"
        "• Operational Driver Metric (D9):\n"
        "  - Mean demand = 33.4 trips/hr. Model MAE = 6.78 trips (~5 drivers).\n"
        "  - Dispatch planning requires an RMSE buffer of ~12 drivers/zone-hour."
    )
    ax.text(0.06, 0.22, error_text, fontsize=8.5, color=TEXT_DARK, va="center", linespacing=1.3)

    # Right Column: Deployment, Live Demo & Recommendations
    draw_card(ax, 0.52, 0.07, 0.44, 0.77, "Live Demo (Deliverable E) & Dispatch Playbook")

    demo_badge = Rectangle((0.54, 0.70), 0.40, 0.07, facecolor="#ecfdf5", edgecolor="#a7f3d0", zorder=3)
    ax.add_patch(demo_badge)
    ax.text(0.55, 0.745, "LIVE DEMO DEPLOYED & TESTED (Deliverable E)", fontsize=9.5, fontweight="bold", color=ACCENT_GREEN, zorder=4)
    ax.text(0.55, 0.715, "URL: https://team-quatro-addis-rides.streamlit.app\nLocal: streamlit run app/app.py", fontsize=8.2, color=TEXT_DARK, zorder=4)

    recs_text = (
        "• Operational Dispatch Playbook:\n"
        "  1. Commuter Staging: Pre-dispatch 60% of idle fleet to Bole, Kazanchis,\n"
        "     and Merkato by 06:30 and 16:30.\n"
        "  2. Weather Surge Alerts: Trigger driver notifications 45 min before\n"
        "     forecasted rain (>2mm) to capture +25% unfulfilled demand.\n"
        "  3. Stadium Dispersal: Position drivers 30 min before matches conclude\n"
        "     at Addis Ababa Stadium using hours_until_event lookup.\n\n"
        "• Stretch Goals & Future Work (Memo):\n"
        "  - Rolling drift detection (PSI > 0.25 trigger retrain).\n"
        "  - Spatial equity: Driver subsidy in peripheral zones (Kolfe, Ayat).\n"
        "  - Real-time GPS telematics integration for supply-side equilibrium."
    )
    ax.text(0.54, 0.42, recs_text, fontsize=8.5, color=TEXT_DARK, va="center", linespacing=1.3)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"[slides] Generating 5-slide PDF presentation at {OUT_PDF}...")
    with PdfPages(OUT_PDF) as pdf:
        build_slide_1(pdf)
        print("  -> Rendered Slide 1: The Problem & The Three Tables")
        build_slide_2(pdf)
        print("  -> Rendered Slide 2: Cleaning, Clock Proof & Integration (Fig 06)")
        build_slide_3(pdf)
        print("  -> Rendered Slide 3: What the Data Says (Fig 04 Heatmap & Fig 07 Rain)")
        build_slide_4(pdf)
        print("  -> Rendered Slide 4: Modeling, Ablation & Rolling-Origin (Fig 10 & 12)")
        build_slide_5(pdf)
        print("  -> Rendered Slide 5: Error Analysis, Demo & Operations Playbook (Fig 11)")
    print(f"[slides] Successfully generated: {OUT_PDF.resolve()} ({OUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
