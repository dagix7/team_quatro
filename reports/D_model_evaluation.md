# Deliverable D - Modeling & Evaluation (team your_team_name)

*Final model:* LightGBM (target mode: `raw`), fitted on 1 Jan - 31 Oct 2025. *Holdout split:* train < 2025-10-18, validate 2025-10-18 - 2025-10-31. *Rolling-origin cuts:* 2025-06-21, 2025-07-19, 2025-08-16, 2025-09-13, 2025-10-18. *Tuning cuts:* 2025-07-05, 2025-08-02, 2025-08-30, 2025-09-27. *Seed:* 42. No score is computed on the test file.

## D1 Baselines

Validation 2025-10-18 to 2025-10-31. The seasonal-naive baseline removes 65% of the mean-predictor RMSE, so most of the signal is the weekly rhythm per zone; everything a model adds must beat this.

| baseline | rmse | mae |
|---|---|---|
| Mean predictor | 27.531 | 20.689 |
| Seasonal-naive (all train) | 11.298 | 7.360 |
| Seasonal-naive (last 8 weeks) | 9.560 | 6.405 |

## D4 Feature availability audit

Only forecast-time-safe columns enter the model. Operational columns are excluded because they are effects of demand, not causes.

| feature | group | source | known_at_forecast_time | reason | decision |
|---|---|---|---|---|---|
| hour | calendar | ts | yes | pure function of the timestamp | used |
| dow | calendar | ts | yes | pure function of the timestamp | used |
| is_weekend | calendar | ts | yes | pure function of the timestamp | used |
| hour_of_week | calendar | ts | yes | pure function of the timestamp | used |
| day_of_month | calendar | ts | yes | pure function of the timestamp | used |
| is_payday_window | calendar | ts | yes | function of day of month | used |
| is_holiday | calendar | events_calendar (scheduled) | yes | official holidays / school terms are published in advance | used |
| is_holiday_eve | calendar | events_calendar (scheduled) | yes | official holidays / school terms are published in advance | used |
| is_school_break | calendar | events_calendar (scheduled) | yes | official holidays / school terms are published in advance | used |
| temp_c | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| rain_mm | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| humidity_pct | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| wind_kmh | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| rain_3h | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| rain_6h | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| rain_24h | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| rain_class | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| rain_missing | weather | weather_hourly | yes (forecast) | Nov rows are *forecasts* (noisier than observed history) - see weather stress test | used |
| ev_football_match_pre | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_football_match_during | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_football_match_post | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_concert_pre | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_concert_during | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_concert_post | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_conference_pre | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_conference_during | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_conference_post | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_exhibition_pre | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_exhibition_during | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_exhibition_post | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_road_closure_pre | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_road_closure_during | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_road_closure_post | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_sports_run_pre | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_sports_run_during | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_sports_run_post | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_n_windows | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_attendance_log | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_hours_to_next | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |
| ev_hours_since_end | events | events_calendar | yes (scheduled) | confirmed events are announced ahead; cancelled ones excluded in A | used |

## D4b Leakage experiment

Adding same-hour driver/wait columns drops RMSE from 9.28 to 3.43 - a fake gain: in production next week's drivers and waits are unknown, so this model could not be run.

| model | rmse | mae |
|---|---|---|
| LightGBM, clean features | 9.279 | 6.110 |
| LightGBM + active_drivers + avg_wait_min (LEAKY) | 3.428 | 2.123 |

## D2 Model comparison

Winner: LightGBM (default). All models use identical features and the same chronological split (2025-10-18 to 2025-10-31).

| model | rmse | mae | train_seconds | rmse_vs_seasonal_naive_% |
|---|---|---|---|---|
| LightGBM (default) | 9.279 | 6.110 | 27.329 | -2.933 |
| HistGradientBoosting | 9.326 | 6.196 | 19.779 | -2.447 |
| MLP (neural net) | 9.464 | 6.451 | 56.637 | -0.998 |
| Random forest | 9.654 | 6.426 | 196.186 | 0.981 |
| Ridge (linear) | 10.285 | 6.924 | 7.400 | 7.588 |

## D5 Ablation

Weather alone changes mean RMSE by -9.0%, events alone by +0.2%, both by -9.4% (negative = better). Means over the 5 rolling folds.

| variant | n_features | rmse_mean | rmse_sd | mae_mean | rmse_holdout_18oct | d_rmse_vs_(i) | d_rmse_% | d_mae_vs_(i) |
|---|---|---|---|---|---|---|---|---|
| (i) calendar + zone + trend | 18 | 9.875 | 1.195 | 6.402 | 9.520 | 0.000 | 0.000 | 0.000 |
| (ii) + weather | 27 | 8.984 | 0.862 | 5.954 | 9.016 | -0.892 | -9.028 | -0.448 |
| (iii) + events | 40 | 9.893 | 1.281 | 6.394 | 9.979 | 0.018 | 0.186 | -0.009 |
| (iv) + weather + events | 49 | 8.945 | 0.994 | 5.920 | 9.279 | -0.930 | -9.416 | -0.483 |

## D6 Tuning summary

Random search, 30 trials over ['objective', 'num_leaves', 'learning_rate', 'n_estimators', 'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda'], time-ordered folds ['2025-07-05', '2025-08-02', '2025-08-30', '2025-09-27']. Best params: {'subsample_freq': 1, 'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'objective': 'regression', 'num_leaves': 31, 'learning_rate': 0.08, 'n_estimators': 900, 'min_child_samples': 20, 'subsample': 1.0, 'colsample_bytree': 0.7, 'reg_lambda': 20.0}

| setting | rmse_holdout | mae_holdout | rmse_tuning_folds |
|---|---|---|---|
| default | 9.279 | 6.110 | 9.198 |
| tuned | 9.200 | 6.072 | 9.051 |

## D6 Tuning top trials

| trial | objective | num_leaves | learning_rate | n_estimators | min_child_samples | subsample | colsample_bytree | reg_lambda | tune_rmse |
|---|---|---|---|---|---|---|---|---|---|
| 10 | regression | 31 | 0.080 | 900 | 20 | 1.000 | 0.700 | 20.000 | 9.051 |
| 29 | poisson | 63 | 0.080 | 900 | 20 | 0.600 | 0.700 | 5.000 | 9.068 |
| 22 | poisson | 127 | 0.080 | 1200 | 80 | 1.000 | 0.900 | 20.000 | 9.104 |
| 11 | poisson | 127 | 0.050 | 900 | 20 | 0.600 | 0.900 | 1.000 | 9.128 |
| 28 | regression | 63 | 0.080 | 900 | 40 | 1.000 | 0.900 | 5.000 | 9.135 |
| 12 | poisson | 63 | 0.030 | 900 | 20 | 0.600 | 0.700 | 5.000 | 9.158 |
| default | regression | 63 | 0.030 | 800 | 40 | 0.800 | 0.800 | 1.000 | 9.198 |
| 7 | regression | 63 | 0.050 | 1200 | 80 | 1.000 | 0.900 | 0.000 | 9.203 |
| 21 | regression | 31 | 0.030 | 900 | 40 | 1.000 | 0.700 | 0.000 | 9.251 |
| 16 | regression | 31 | 0.080 | 1200 | 20 | 0.800 | 0.500 | 5.000 | 9.259 |

## D3 Rolling-origin validation

Final model RMSE 8.80 ± 0.93 vs seasonal-naive 9.89 ± 1.06 (mean ± sd over 5 folds). Worst fold: 2025-09-13.

| cut | n_val | mean_trips | rmse_final | mae_final | rmse_sn | mae_sn | improvement_% |
|---|---|---|---|---|---|---|---|
| 2025-06-21 | 3924 | 29.715 | 8.114 | 5.621 | 9.044 | 5.984 | 10.287 |
| 2025-07-19 | 3921 | 31.200 | 8.422 | 5.718 | 10.192 | 6.571 | 17.369 |
| 2025-08-16 | 3941 | 31.433 | 8.017 | 5.622 | 9.063 | 6.081 | 11.547 |
| 2025-09-13 | 3920 | 32.967 | 10.228 | 6.239 | 11.582 | 6.964 | 11.691 |
| 2025-10-18 | 3914 | 33.448 | 9.200 | 6.072 | 9.560 | 6.405 | 3.758 |

## D7a Error by zone

| zone | n | mean_trips | rmse | mae | bias(pred-actual) | rmse_%_of_mean |
|---|---|---|---|---|---|---|
| Arat Kilo | 1639 | 23.525 | 6.688 | 4.627 | -0.139 | 28.431 |
| Ayat | 1641 | 20.068 | 5.494 | 4.031 | -0.374 | 27.376 |
| Bole | 1629 | 43.612 | 10.774 | 7.908 | -0.321 | 24.704 |
| CMC | 1644 | 28.451 | 7.532 | 5.449 | -0.170 | 26.473 |
| Gerji | 1628 | 27.051 | 6.885 | 5.042 | -0.315 | 25.451 |
| Kazanchis | 1630 | 36.452 | 13.030 | 7.223 | -0.779 | 35.746 |
| Kolfe | 1638 | 25.768 | 6.898 | 4.980 | -0.558 | 26.768 |
| Lideta | 1629 | 32.042 | 8.077 | 5.679 | -1.179 | 25.207 |
| Megenagna | 1636 | 43.085 | 10.435 | 7.412 | 0.158 | 24.220 |
| Merkato | 1634 | 44.571 | 11.099 | 7.379 | -0.953 | 24.902 |
| Piassa | 1637 | 32.557 | 9.464 | 6.044 | -1.047 | 29.068 |
| Sarbet | 1635 | 23.964 | 6.146 | 4.494 | -0.204 | 25.646 |

## D7a Error by hour

| hour | n | mean_trips | rmse | mae | bias(pred-actual) | rmse_%_of_mean |
|---|---|---|---|---|---|---|
| 0.000 | 822.000 | 9.052 | 4.653 | 2.714 | -0.099 | 51.397 |
| 1.000 | 817.000 | 7.118 | 3.392 | 2.271 | 0.064 | 47.650 |
| 2.000 | 819.000 | 6.169 | 2.908 | 2.111 | 0.107 | 47.139 |
| 3.000 | 821.000 | 6.440 | 2.772 | 2.085 | -0.071 | 43.047 |
| 4.000 | 812.000 | 8.717 | 3.547 | 2.602 | 0.110 | 40.695 |
| 5.000 | 813.000 | 17.072 | 5.155 | 3.755 | -0.269 | 30.197 |
| 6.000 | 824.000 | 31.168 | 8.273 | 6.096 | -0.306 | 26.545 |
| 7.000 | 811.000 | 47.300 | 10.950 | 8.046 | -1.779 | 23.149 |
| 8.000 | 815.000 | 48.242 | 10.217 | 7.566 | -1.472 | 21.179 |
| 9.000 | 820.000 | 38.454 | 8.629 | 6.526 | -0.545 | 22.440 |
| 10.000 | 812.000 | 32.541 | 7.709 | 5.710 | -0.610 | 23.691 |
| 11.000 | 816.000 | 34.001 | 8.425 | 6.220 | -0.315 | 24.778 |
| 12.000 | 809.000 | 38.485 | 8.793 | 6.608 | -0.840 | 22.848 |
| 13.000 | 821.000 | 41.786 | 10.449 | 7.464 | -0.571 | 25.007 |
| 14.000 | 820.000 | 40.418 | 9.964 | 7.379 | -0.306 | 24.653 |
| 15.000 | 825.000 | 38.262 | 9.414 | 7.040 | 0.714 | 24.604 |
| 16.000 | 812.000 | 44.082 | 10.550 | 7.720 | 0.202 | 23.932 |
| 17.000 | 812.000 | 55.599 | 15.259 | 9.984 | -1.247 | 27.446 |
| 18.000 | 826.000 | 59.690 | 14.608 | 9.708 | -1.638 | 24.474 |
| 19.000 | 820.000 | 54.297 | 11.740 | 8.619 | -0.727 | 21.622 |
| 20.000 | 817.000 | 42.154 | 9.522 | 7.157 | -0.855 | 22.588 |
| 21.000 | 817.000 | 28.845 | 7.910 | 5.649 | -0.499 | 27.422 |
| 22.000 | 821.000 | 19.734 | 6.106 | 4.227 | -0.540 | 30.943 |
| 23.000 | 818.000 | 12.491 | 5.144 | 3.250 | -0.275 | 41.183 |

## D7a Error by day type

| day_type | n | mean_trips | rmse | mae | bias(pred-actual) | rmse_%_of_mean |
|---|---|---|---|---|---|---|
| weekday | 14020 | 33.079 | 9.009 | 5.961 | -0.635 | 27.234 |
| weekend | 5600 | 28.426 | 8.380 | 5.587 | -0.125 | 29.481 |

## D7a Error by event presence

| has_event | n | mean_trips | rmse | mae | bias(pred-actual) | rmse_%_of_mean |
|---|---|---|---|---|---|---|
| 0.000 | 19113.000 | 31.352 | 8.627 | 5.758 | -0.520 | 27.517 |
| 1.000 | 507.000 | 46.803 | 14.633 | 9.489 | 0.657 | 31.264 |

## D7b Ten largest errors

Hypotheses to be written by the team after inspection (see notebook).

| zone | ts | trips | pred | err | day_type | rain_mm | active_event_windows | unexplained_by_calendar | hypothesis |
|---|---|---|---|---|---|---|---|---|---|
| Kazanchis | 2025-09-26 17:00:00 | 311.000 | 129.600 | -181.400 | weekday | 3.400 | none | YES - no listed event | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Kazanchis | 2025-09-26 18:00:00 | 321.000 | 148.500 | -172.500 | weekday | 4.900 | none | YES - no listed event | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Kazanchis | 2025-10-18 17:00:00 | 36.000 | 142.900 | 106.900 | weekend | 0.000 | football_match_post, n_windows | no | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Kazanchis | 2025-08-01 18:00:00 | 188.000 | 101.000 | -87.000 | weekday | 3.300 | none | YES - no listed event | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Piassa | 2025-09-26 17:00:00 | 198.000 | 112.400 | -85.600 | weekday | 3.400 | none | YES - no listed event | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Kazanchis | 2025-09-24 21:00:00 | 47.000 | 129.900 | 82.900 | weekday | 0.000 | football_match_post, n_windows | no | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Kazanchis | 2025-10-25 17:00:00 | 67.000 | 148.900 | 81.900 | weekend | 0.000 | football_match_post, n_windows | no | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Piassa | 2025-09-25 17:00:00 | 153.000 | 72.900 | -80.100 | weekday | 9.700 | none | YES - no listed event | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Megenagna | 2025-10-29 18:00:00 | 183.000 | 112.700 | -70.300 | weekday | 3.500 | none | YES - no listed event | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |
| Kazanchis | 2025-10-25 18:00:00 | 58.000 | 126.000 | 68.000 | weekend | 0.000 | football_match_post, n_windows, hours_since_end | no | TODO: write after looking at the row (unlisted event? rain? holiday? outage?) |

## D8 Response to findings

Ratio-target experiment: RMSE 8.796 -> 8.804. Adopted: False.

| variant | rmse_mean | rmse_sd | mae_mean |
|---|---|---|---|
| raw trips target (current) | 8.796 | 0.926 | 5.854 |
| ratio target (trips / level_28d) | 8.804 | 0.837 | 5.878 |

## D8 Per-zone effect of ratio target

| zone | rmse_raw | rmse_ratio | change_% |
|---|---|---|---|
| Arat Kilo | 6.688 | 6.548 | -2.099 |
| Ayat | 5.494 | 5.535 | 0.741 |
| Bole | 10.774 | 10.997 | 2.072 |
| CMC | 7.532 | 7.578 | 0.614 |
| Gerji | 6.885 | 6.910 | 0.370 |
| Kazanchis | 13.030 | 12.632 | -3.055 |
| Kolfe | 6.898 | 6.855 | -0.613 |
| Lideta | 8.077 | 8.193 | 1.443 |
| Megenagna | 10.435 | 10.489 | 0.521 |
| Merkato | 11.099 | 11.392 | 2.641 |
| Piassa | 9.464 | 9.298 | -1.746 |
| Sarbet | 6.146 | 6.146 | 0.003 |

## D9 Plain-language metric

On the 18-31 Oct holdout the model is typically off by 6.1 trips per zone-hour (MAE) and its RMSE is 9.2 trips, i.e. 18% and 28% of the average demand of 33.4 trips per zone-hour. At about 1.3 trips per driver-hour, that is roughly 4.7 drivers (MAE) or 7.1 drivers (RMSE) per zone-hour. Across 5 rolling folds RMSE is 8.8 ± 0.9. Practically: plan driver supply per zone-hour with a buffer of about 8 drivers, and expect larger misses at event hours.

| metric | trips_per_zone_hour | pct_of_mean_demand | approx_drivers |
|---|---|---|---|
| RMSE | 9.200 | 27.507 | 7.077 |
| MAE | 6.072 | 18.155 | 4.671 |

## Extra weather forecast stress test

Noise levels are assumptions, not measured forecast error; treat as a sensitivity check.

| weather | rmse | mae |
|---|---|---|
| observed (clean) | 9.200 | 6.072 |
| noisy forecast-like | 9.267 | 6.097 |

## D-fig12 Permutation importance

| feature | importance | sd | group |
|---|---|---|---|
| same_how_mean | 12.991 | 0.172 | Trend / lag |
| hour | 6.097 | 0.102 | Calendar |
| zone | 3.183 | 0.128 | Zone |
| level_28d | 0.958 | 0.042 | Trend / lag |
| dow | 0.727 | 0.036 | Calendar |
| hour_of_week | 0.710 | 0.040 | Calendar |
| rain_mm | 0.485 | 0.024 | Weather |
| lag_672h | 0.296 | 0.028 | Trend / lag |
| lag_504h | 0.282 | 0.043 | Trend / lag |
| level_7d | 0.176 | 0.040 | Trend / lag |
| lag_336h | 0.156 | 0.023 | Trend / lag |
| temp_c | 0.086 | 0.028 | Weather |
| is_weekend | 0.046 | 0.008 | Calendar |
| rain_class | 0.044 | 0.006 | Weather |
| ev_conference_during | 0.029 | 0.004 | Events / calendar-from-events |
| is_payday_window | 0.029 | 0.005 | Calendar |
| rain_6h | 0.028 | 0.009 | Weather |
| ev_concert_post | 0.027 | 0.007 | Events / calendar-from-events |
| humidity_pct | 0.026 | 0.013 | Weather |
| day_of_month | 0.023 | 0.020 | Calendar |
| ev_road_closure_during | 0.022 | 0.009 | Events / calendar-from-events |
| rain_24h | 0.017 | 0.016 | Weather |
| ev_concert_pre | 0.015 | 0.002 | Events / calendar-from-events |
| ev_football_match_during | 0.007 | 0.003 | Events / calendar-from-events |
| ev_conference_post | 0.001 | 0.001 | Events / calendar-from-events |
| ev_road_closure_post | 0.000 | 0.001 | Events / calendar-from-events |
| is_school_break | 0.000 | 0.000 | Events / calendar-from-events |
| is_holiday | 0.000 | 0.000 | Events / calendar-from-events |
| is_holiday_eve | 0.000 | 0.000 | Events / calendar-from-events |
| ev_sports_run_during | 0.000 | 0.000 | Events / calendar-from-events |
| ev_road_closure_pre | 0.000 | 0.000 | Events / calendar-from-events |
| ev_exhibition_post | 0.000 | 0.000 | Events / calendar-from-events |
| ev_exhibition_pre | 0.000 | 0.000 | Events / calendar-from-events |
| ev_sports_run_post | 0.000 | 0.000 | Events / calendar-from-events |
| ev_sports_run_pre | 0.000 | 0.000 | Events / calendar-from-events |
| ev_conference_pre | 0.000 | 0.000 | Events / calendar-from-events |
| rain_3h | -0.000 | 0.003 | Weather |
| ev_concert_during | -0.000 | 0.000 | Events / calendar-from-events |
| ev_n_windows | -0.001 | 0.002 | Events / calendar-from-events |
| ev_football_match_post | -0.001 | 0.000 | Events / calendar-from-events |

## Team interpretation (fill in)

_Copy your interpretations from the notebook's ✍️ cells here._
