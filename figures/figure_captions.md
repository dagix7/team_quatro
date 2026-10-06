# Figure Captions

## fig01_gaps_and_missingness.png
Side-by-side bar charts showing missing timestamp hours and consecutive zero-trip runs per zone across the training period (Jan–Oct 2025). Zones average 3.15% missing slots; the pattern confirms that lag-feature imputation is required before modelling and that overnight zero-trip hours are structural rather than anomalous.

## fig02_before_after_cleaning.png
Ranked horizontal bar chart comparing total trip volume per zone before and after the data-cleaning pipeline (negative removal, ×8 corruption fix, duplicate averaging). Merkato leads the cleaned dataset with 285,648 trips — 3× the volume of the lowest zone Ayat (95,486) — confirming that post-cleaning demand signals are well-differentiated across zones.

## fig03_demand_trend_with_holidays.png
Daily aggregate ride requests over the full training timeline with public holidays annotated. The series peaks at 11,751 trips on 2025-09-26 and shows a gradual upward trend with weekly seasonality; holiday dates are visible as demand dips, validating the `is_holiday` feature as a meaningful temporal signal.

## fig04_hour_by_weekday_heatmap.png
24×7 heatmap of mean hourly trips (rows = hours 0–23, columns = Mon–Sun). The evening rush window 17:00–19:00 on weekdays is the hottest band, with Friday 18:00 as the single peak cell — providing a direct lookup table for baseline dispatch-level planning by day and hour.

## fig05_zone_profiles.png
Overlaid diurnal demand curves for the three highest-volume zones (Merkato, Bole, Megenagna) and three lowest-volume zones (Ayat, Arat Kilo, Sarbet). High-density commercial hubs show sharp morning and evening twin peaks while suburban zones are flat and low throughout the day — a key input for zone-specific fleet staging windows.

## fig06_weather_timezone_check.png
Pearson correlation matrix confirming that all weather variables are aligned to EAT (UTC+3) before joining. Temperature shows the strongest positive association with trips (r = 0.383); rain shows a moderate positive correlation (r = 0.200); humidity and wind are near-zero, validating that only temperature and rain carry actionable dispatch signal.

## fig07_rain_effect.png
Grouped bar chart comparing mean trips per zone-hour during clear conditions vs rainy hours (rain_mm > 0). The average demand lift under rain is +85% and the pattern is consistent across all 12 zones, confirming that a binary rain flag should immediately trigger city-wide surge dispatch regardless of zone.

## fig08_event_study.png
Event-study chart showing mean hourly demand relative to a no-event baseline across all major event types (football matches, concerts, sports runs, road closures). Concert windows drive the largest surge (+143.7% lift), followed by football matches (+98%); road closures and exhibitions show moderate suppression effects.

## fig09_holiday_effects.png
Boxplot comparing hourly trip distributions on public holidays versus regular weekdays. Mean demand on public holidays is 23.6 trips/zone-hour vs 28.7 on regular weekdays (−17.6%), with the distribution shifted toward midday — operators should reduce but re-time fleet allocation rather than simply cutting staffing levels.

## fig10_model_comparison.png
Comparison of validation-set RMSE and MAE across candidate model architectures (baseline lag model, LightGBM, XGBoost, RandomForest). The gradient-boosting models outperform the baseline by approximately 18–22% on RMSE, providing the quantitative justification for the final model selection in Deliverable D.

## fig11_forecast_vs_actual.png
Zoomed time-series overlay of predicted versus actual trip counts for a representative two-week holdout window across selected zones. Forecast errors are largest at peak-hour spikes and during the one confirmed 3-hour system outage; the model tracks the weekly seasonality pattern closely outside of event windows.

## fig12_feature_importance.png
Horizontal bar chart of the top 15 RandomForest feature importances trained on a 50,000-row sample of master_train. The `same_how_mean` lag (mean of 14/21/28-day same-hour-of-week lags) dominates at 0.919 importance — confirming that same-week historical demand is by far the strongest predictor — while `is_holiday` (0.014) and `rain_mm` (0.013) are the leading non-lag contextual signals.
