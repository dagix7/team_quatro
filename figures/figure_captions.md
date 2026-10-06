# Figure Captions

## fig01_gaps_and_missingness.png
Missing timestamp hours and consecutive zero-trip runs per zone across the training period. Zones with elevated missing hours (averaging 3.15% of expected slots) require careful lag-feature imputation before modelling; zones with many zero-trip hours may indicate overnight service suspensions rather than genuine demand absences.

## fig02_total_demand_by_zone.png
Horizontal bar chart ranking all 12 zones by cumulative trip volume over the training period (Jan–Oct 2025). Merkato leads with 285,648 trips, more than 3× the lowest-volume zone Ayat (95,486 trips), confirming that fleet allocation strategies must be heavily zone-weighted.

## fig03_hourly_demand_trend.png
Total daily ride requests aggregated across all zones over the full training timeline. The single-day peak of 11,751 trips occurred on 2025-09-26; the series shows a gradual upward trend from January with clear weekly seasonality, indicating the model needs both long-range trend features and weekly lag terms.

## fig04_hour_by_weekday_heatmap.png
24×7 heatmap of mean hourly trips (rows = hour 0–23, columns = Mon–Sun). The brightest cells cluster around the evening rush (hour 17–19) on weekdays, with Friday 18:00 being the single hottest cell, providing a direct lookup table for baseline dispatch planning.

## fig05_zone_hourly_profiles.png
Overlaid diurnal demand curves for the three highest-volume zones (Merkato, Bole, Megenagna) and three lowest-volume zones (Ayat, Arat Kilo, Sarbet). High-density hubs exhibit sharp morning and evening twin peaks while suburban zones show flatter, lower profiles — a key input for zone-specific fleet staging windows.

## fig06_weather_correlation_matrix.png
Pearson correlation matrix between rain_mm, temp_c, humidity_pct, wind_kmh, and trips. Temperature shows the strongest positive association with demand (r = 0.383) and rain shows a moderate positive correlation (r = 0.200); humidity and wind have near-zero correlations, suggesting they add marginal predictive value beyond rain and temperature.

## fig07_rain_vs_demand_by_zone.png
Grouped bar chart comparing mean trips during clear conditions vs rainy hours (rain_mm > 0) for all 12 zones. The average demand lift under rain is +85%, and the pattern is consistent across all zones, confirming that real-time rain status should trigger immediate surge-dispatch signals in every zone.

## fig08_holiday_demand_shift.png
Boxplot of hourly trip distributions on public holidays vs regular weekdays (weekend rows excluded). Mean demand drops by −17.6% on public holidays (23.6 vs 28.7 mean trips), but the distributions overlap considerably — demand does not collapse, it simply shifts later in the day, requiring adjusted rather than reduced staffing.

## fig09_football_event_window.png
Bar chart of average demand across the three football match windows (pre-match −2h, during, post-match +3h) compared to a no-event baseline. The combined match period drives a +98% demand lift over baseline, with the post-match window typically showing the sharpest single-hour spike as crowds disperse.

## fig10_event_type_impact.png
Boxplot comparing per-hour trip lift across Sports events (+57.3%), Concerts (+143.7%), and Public Holidays (−15.9%) relative to a no-event baseline. Concerts are the highest-impact event type and should trigger the largest pre-positioned fleet response; public holidays act as a mild suppressor and should reduce default baseline targets.

## fig11_residual_outages.png
Zoomed time-series showing confirmed system outage windows — sequences of ≥3 consecutive zero-trip hours that are inconsistent with normal overnight lows. Only one outage period totalling 3 hours was detected in the dataset, meaning the cleaned master_train is largely free of multi-hour service interruptions.

## fig12_feature_importance.png
Horizontal bar chart of the top 15 RandomForest feature importances trained on a 50,000-row sample of master_train. The `same_how_mean` lag (mean of 14/21/28-day same-hour lags) dominates at 0.919 importance, confirming that same-hour-of-week historical demand is by far the strongest predictor; `is_holiday` (0.014) and `rain_mm` (0.013) are the leading non-lag contextual features.
