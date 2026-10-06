"""
Generate all 12 figures and compute 14 analytical metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style and color palettes
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Define paths
figures_dir = Path('figures')
figures_dir.mkdir(exist_ok=True)

# Dictionary to store metrics
metrics = {}

print('Loading data...')
master_train = pd.read_csv('data/processed/master_train.csv')
raw_train = pd.read_csv('data/raw/ride_demand_train.csv')

# Parse timestamps
master_train['ts'] = pd.to_datetime(master_train['ts'])
raw_train['pickup_hour'] = pd.to_datetime(raw_train['pickup_hour'], format='mixed', dayfirst=True)

print(f'Master train shape: {master_train.shape}')
print(f'Raw train shape: {raw_train.shape}')
print(f'Date range: {master_train.ts.min()} to {master_train.ts.max()}')


# =============================================================================
# FIGURE 1: Gaps and Missingness
# =============================================================================
print('\n[1/12] Generating gaps and missingness figure...')

# Analyze gaps by zone
zones = sorted(master_train['zone'].unique())
gap_stats = []

for zone in zones:
    zone_data = master_train[master_train['zone'] == zone].sort_values('ts')
    
    # Expected hourly timestamps
    expected_range = pd.date_range(
        start=zone_data['ts'].min(),
        end=zone_data['ts'].max(),
        freq='H'
    )
    
    # Missing timestamps
    missing_count = len(expected_range) - len(zone_data)
    missing_pct = (missing_count / len(expected_range)) * 100
    
    # Zero-trip hours
    zero_trips = (zone_data['trips'] == 0).sum()
    zero_pct = (zero_trips / len(zone_data)) * 100
    
    gap_stats.append({
        'zone': zone,
        'missing_hours': missing_count,
        'missing_pct': missing_pct,
        'zero_trips': zero_trips,
        'zero_pct': zero_pct
    })

gap_df = pd.DataFrame(gap_stats)
metrics['total_missing_hours'] = int(gap_df['missing_hours'].sum())
metrics['avg_missing_pct'] = float(gap_df['missing_pct'].mean())
metrics['total_zero_trips'] = int(gap_df['zero_trips'].sum())

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Missing hours
axes[0].barh(gap_df['zone'], gap_df['missing_hours'], color=sns.color_palette('viridis', len(gap_df)))
axes[0].set_xlabel('Missing Hours')
axes[0].set_ylabel('Zone')
axes[0].set_title('Missing Timestamp Hours by Zone')
axes[0].grid(axis='x', alpha=0.3)

# Zero-trip runs
axes[1].barh(gap_df['zone'], gap_df['zero_trips'], color=sns.color_palette('mako', len(gap_df)))
axes[1].set_xlabel('Zero-Trip Hours')
axes[1].set_ylabel('Zone')
axes[1].set_title('Zero-Trip Hours by Zone')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'fig01_gaps_and_missingness.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig01_gaps_and_missingness.png saved')


# =============================================================================
# FIGURE 2: Total Demand by Zone
# =============================================================================
print('[2/12] Generating total demand by zone...')

zone_totals = master_train.groupby('zone')['trips'].sum().sort_values()
metrics['top_zone'] = zone_totals.idxmax()
metrics['top_zone_trips'] = int(zone_totals.max())
metrics['zone_demand_ranking'] = zone_totals.to_dict()

fig, ax = plt.subplots(figsize=(10, 8))
colors = sns.color_palette('YlGnBu', len(zone_totals))
zone_totals.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Total Trips')
ax.set_ylabel('Zone')
ax.set_title('Total Ride Demand by Zone (Ranked)')
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(zone_totals.values):
    ax.text(v + zone_totals.max()*0.01, i, f'{int(v):,}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(figures_dir / 'fig02_total_demand_by_zone.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig02_total_demand_by_zone.png saved')


# =============================================================================
# FIGURE 3: Hourly Demand Trend
# =============================================================================
print('[3/12] Generating hourly demand trend...')

daily_demand = master_train.groupby(master_train['ts'].dt.date)['trips'].sum()
metrics['avg_daily_trips'] = float(daily_demand.mean())
metrics['peak_daily_trips'] = int(daily_demand.max())
metrics['peak_day'] = str(daily_demand.idxmax())

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_demand.index, daily_demand.values, linewidth=1.5, color='#2E86AB')
ax.fill_between(daily_demand.index, daily_demand.values, alpha=0.3, color='#2E86AB')
ax.set_xlabel('Date')
ax.set_ylabel('Total Daily Trips')
ax.set_title('Daily Ride Demand Trend Over Training Period')
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(figures_dir / 'fig03_hourly_demand_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig03_hourly_demand_trend.png saved')


# =============================================================================
# FIGURE 4: Hour by Weekday Heatmap
# =============================================================================
print('[4/12] Generating hour by weekday heatmap...')

# Create pivot table
heatmap_data = master_train.pivot_table(
    values='trips',
    index='hour',
    columns='dow',
    aggfunc='mean'
)

# Day names
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
heatmap_data.columns = day_names

metrics['peak_hour'] = int(heatmap_data.mean(axis=1).idxmax())
metrics['peak_weekday'] = heatmap_data.mean(axis=0).idxmax()

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(heatmap_data, annot=False, fmt='.0f', cmap='viridis', 
            cbar_kws={'label': 'Average Trips'}, ax=ax)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Hour of Day')
ax.set_title('Average Trips: Hour vs Day of Week Heatmap')
plt.tight_layout()
plt.savefig(figures_dir / 'fig04_hour_by_weekday_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig04_hour_by_weekday_heatmap.png saved')


# =============================================================================
# FIGURE 5: Zone Hourly Profiles
# =============================================================================
print('[5/12] Generating zone hourly profiles...')

# Select diverse zones (top 3 and bottom 3 by volume)
top_zones = zone_totals.tail(3).index.tolist()
bottom_zones = zone_totals.head(3).index.tolist()
selected_zones = top_zones + bottom_zones

hourly_profiles = master_train[master_train['zone'].isin(selected_zones)].groupby(['zone', 'hour'])['trips'].mean().unstack(level=0)

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette('cividis', len(selected_zones))

for i, zone in enumerate(selected_zones):
    label_suffix = '(high)' if zone in top_zones else '(low)'
    ax.plot(hourly_profiles.index, hourly_profiles[zone], 
            marker='o', markersize=3, linewidth=2, 
            label=f'{zone} {label_suffix}', color=colors[i])

ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Trips')
ax.set_title('Diurnal Demand Profiles: High vs Low Volume Zones')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'fig05_zone_hourly_profiles.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig05_zone_hourly_profiles.png saved')


# =============================================================================
# FIGURE 6: Weather Correlation Matrix
# =============================================================================
print('[6/12] Generating weather correlation matrix...')

weather_cols = ['rain_mm', 'temp_c', 'humidity_pct', 'wind_kmh', 'trips']
corr_data = master_train[weather_cols].corr()

# Store key correlations
metrics['rain_trips_correlation'] = float(corr_data.loc['rain_mm', 'trips'])
metrics['temp_trips_correlation'] = float(corr_data.loc['temp_c', 'trips'])

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1,
            cbar_kws={'label': 'Pearson Correlation'}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Weather Variables vs Trips: Correlation Matrix')
plt.tight_layout()
plt.savefig(figures_dir / 'fig06_weather_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig06_weather_correlation_matrix.png saved')


# =============================================================================
# FIGURE 7: Rain vs Demand by Zone
# =============================================================================
print('[7/12] Generating rain vs demand comparison...')

master_train['rain_condition'] = master_train['rain_mm'].apply(lambda x: 'Rain' if x > 0 else 'Clear')

rain_comparison = master_train.groupby(['zone', 'rain_condition'])['trips'].mean().unstack(fill_value=0)

# Calculate lift
rain_comparison['lift_pct'] = ((rain_comparison['Rain'] - rain_comparison['Clear']) / rain_comparison['Clear'] * 100)
metrics['avg_rain_lift_pct'] = float(rain_comparison['lift_pct'].mean())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(rain_comparison.index))
width = 0.35

ax.bar(x - width/2, rain_comparison['Clear'], width, label='Clear', color='#F4A261')
ax.bar(x + width/2, rain_comparison['Rain'], width, label='Rain', color='#2A9D8F')

ax.set_xlabel('Zone')
ax.set_ylabel('Average Trips')
ax.set_title('Average Demand: Clear Weather vs Rain by Zone')
ax.set_xticks(x)
ax.set_xticklabels(rain_comparison.index, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'fig07_rain_vs_demand_by_zone.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig07_rain_vs_demand_by_zone.png saved')


# =============================================================================
# FIGURE 8: Holiday Demand Shift
# =============================================================================
print('[8/12] Generating holiday demand shift...')

holiday_comparison = master_train[master_train['is_weekend'] == 0].groupby('is_holiday')['trips'].apply(list)

# Calculate statistics
metrics['avg_holiday_trips'] = float(master_train[master_train['is_holiday'] == 1]['trips'].mean())
metrics['avg_regular_trips'] = float(master_train[master_train['is_holiday'] == 0]['trips'].mean())
metrics['holiday_demand_change_pct'] = float(
    ((metrics['avg_holiday_trips'] - metrics['avg_regular_trips']) / metrics['avg_regular_trips']) * 100
)

fig, ax = plt.subplots(figsize=(10, 6))
box_data = [
    master_train[master_train['is_holiday'] == 0]['trips'].values,
    master_train[master_train['is_holiday'] == 1]['trips'].values
]

bp = ax.boxplot(box_data, labels=['Regular Weekdays', 'Public Holidays'],
                patch_artist=True, showmeans=True)

colors = ['#E76F51', '#F4A261']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Trips per Hour')
ax.set_title('Demand Distribution: Regular Weekdays vs Public Holidays')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'fig08_holiday_demand_shift.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig08_holiday_demand_shift.png saved')


# =============================================================================
# FIGURE 9: Football Event Window
# =============================================================================
print('[9/12] Generating football event window analysis...')

# Filter football match periods
football_cols = ['ev_football_match_pre', 'ev_football_match_during', 'ev_football_match_post']
football_data = master_train[master_train[football_cols].sum(axis=1) > 0].copy()

if len(football_data) > 0:
    # Create time window relative to match
    football_data['window_type'] = 'Other'
    football_data.loc[football_data['ev_football_match_pre'] == 1, 'window_type'] = 'Pre-Match (-3 to 0h)'
    football_data.loc[football_data['ev_football_match_during'] == 1, 'window_type'] = 'During Match'
    football_data.loc[football_data['ev_football_match_post'] == 1, 'window_type'] = 'Post-Match (0 to +3h)'
    
    window_demand = football_data.groupby('window_type')['trips'].mean().sort_index()
    baseline_demand = master_train[master_train[football_cols].sum(axis=1) == 0]['trips'].mean()
    
    metrics['football_match_lift_pct'] = float(
        ((window_demand.mean() - baseline_demand) / baseline_demand) * 100
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E63946', '#F77F00', '#06AED5']
    window_demand.plot(kind='bar', ax=ax, color=colors)
    ax.axhline(baseline_demand, color='gray', linestyle='--', linewidth=2, label='Baseline (No Event)')
    ax.set_xlabel('Event Window')
    ax.set_ylabel('Average Trips')
    ax.set_title('Demand Surge Around Football Matches')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
else:
    metrics['football_match_lift_pct'] = 0.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No football match data available', 
            ha='center', va='center', fontsize=14)
    ax.set_title('Demand Surge Around Football Matches')

plt.tight_layout()
plt.savefig(figures_dir / 'fig09_football_event_window.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig09_football_event_window.png saved')


# =============================================================================
# FIGURE 10: Event Type Impact
# =============================================================================
print('[10/12] Generating event type impact comparison...')

# Calculate baseline (no events)
baseline_trips = master_train[master_train['ev_n_windows'] == 0]['trips'].mean()

# Analyze different event types
event_types = {
    'Sports': ['ev_football_match_during', 'ev_sports_run_during'],
    'Concerts': ['ev_concert_during'],
    'Holidays': ['is_holiday']
}

event_impact = []
for event_name, cols in event_types.items():
    event_mask = master_train[cols].sum(axis=1) > 0
    if event_mask.sum() > 0:
        avg_trips = master_train[event_mask]['trips'].mean()
        lift_pct = ((avg_trips - baseline_trips) / baseline_trips) * 100
        event_impact.append({
            'event_type': event_name,
            'lift_pct': lift_pct,
            'trips': master_train[event_mask]['trips'].values
        })

metrics['event_type_impacts'] = {e['event_type']: float(e['lift_pct']) for e in event_impact}

fig, ax = plt.subplots(figsize=(10, 6))

if event_impact:
    box_data = [e['trips'] for e in event_impact]
    labels = [e['event_type'] for e in event_impact]
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True)
    colors = sns.color_palette('mako', len(event_impact))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(baseline_trips, color='gray', linestyle='--', linewidth=2, label='Baseline')
    ax.set_ylabel('Trips per Hour')
    ax.set_title('Demand Impact by Event Type')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No event data available', 
            ha='center', va='center', fontsize=14)
    ax.set_title('Demand Impact by Event Type')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'fig10_event_type_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig10_event_type_impact.png saved')


# =============================================================================
# FIGURE 11: Residual Outages
# =============================================================================
print('[11/12] Generating residual outages visualization...')

# Identify zero-trip anomalies (consecutive zero hours)
master_train_sorted = master_train.sort_values(['zone', 'ts'])
master_train_sorted['is_zero'] = (master_train_sorted['trips'] == 0).astype(int)

# Find consecutive zero sequences
outage_periods = []
for zone in zones:
    zone_data = master_train_sorted[master_train_sorted['zone'] == zone].copy()
    zone_data['zero_group'] = (zone_data['is_zero'] != zone_data['is_zero'].shift()).cumsum()
    
    zero_sequences = zone_data[zone_data['is_zero'] == 1].groupby('zero_group').agg({
        'ts': ['first', 'last', 'count']
    })
    zero_sequences.columns = ['start', 'end', 'duration']
    
    # Filter sequences longer than 3 hours (likely outages, not just low demand)
    long_outages = zero_sequences[zero_sequences['duration'] >= 3]
    
    for idx, row in long_outages.iterrows():
        outage_periods.append({
            'zone': zone,
            'start': row['start'],
            'end': row['end'],
            'duration_hours': row['duration']
        })

metrics['total_outage_periods'] = len(outage_periods)
metrics['total_outage_hours'] = sum([o['duration_hours'] for o in outage_periods]) if outage_periods else 0

# Visualize a sample of outages
fig, ax = plt.subplots(figsize=(14, 6))

if outage_periods:
    # Take first 5 significant outages
    sample_outages = sorted(outage_periods, key=lambda x: x['duration_hours'], reverse=True)[:5]
    
    for i, outage in enumerate(sample_outages):
        zone = outage['zone']
        start = outage['start']
        end = outage['end']
        
        # Get data around the outage (±12 hours)
        context_start = start - pd.Timedelta(hours=12)
        context_end = end + pd.Timedelta(hours=12)
        
        context_data = master_train[
            (master_train['zone'] == zone) & 
            (master_train['ts'] >= context_start) & 
            (master_train['ts'] <= context_end)
        ].sort_values('ts')
        
        if len(context_data) > 0:
            ax.plot(context_data['ts'], context_data['trips'], 
                   label=f'{zone} ({outage["duration_hours"]}h outage)', 
                   marker='o', markersize=3, linewidth=1.5, alpha=0.7)
            
            # Highlight outage period
            ax.axvspan(start, end, alpha=0.2, color='red')
    
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Trips')
    ax.set_title('System Outages and Zero-Trip Anomalies (Top 5 by Duration)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
else:
    ax.text(0.5, 0.5, 'No significant outage periods detected', 
            ha='center', va='center', fontsize=14)
    ax.set_title('System Outages and Zero-Trip Anomalies')

plt.tight_layout()
plt.savefig(figures_dir / 'fig11_residual_outages.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig11_residual_outages.png saved')


# =============================================================================
# FIGURE 12: Feature Importance
# =============================================================================
print('[12/12] Generating feature importance...')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Prepare features for modeling
    feature_cols = [
        'hour', 'dow', 'is_weekend', 'day_of_month', 'month',
        'is_payday_window', 'is_holiday', 'is_holiday_eve', 'is_school_break',
        'temp_c', 'rain_mm', 'humidity_pct', 'wind_kmh',
        'rain_3h', 'rain_6h', 'rain_24h', 'rain_class',
        'ev_football_match_pre', 'ev_football_match_during', 'ev_football_match_post',
        'ev_concert_pre', 'ev_concert_during', 'ev_concert_post',
        'ev_n_windows', 'ev_attendance_log',
        'lag_336h', 'lag_504h', 'lag_672h', 'same_how_mean',
        'level_28d', 'level_7d', 'level_ratio_7_28'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in master_train.columns]
    
    # Prepare data (use a sample for speed)
    sample_size = min(50000, len(master_train))
    sample_data = master_train.sample(n=sample_size, random_state=42)
    
    X = sample_data[available_features].fillna(0)
    y = sample_data['trips']
    
    # Train quick model
    print('  Training RandomForest for feature importance...')
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    
    metrics['top_features'] = importances.set_index('feature')['importance'].to_dict()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette('viridis', len(importances))
    ax.barh(importances['feature'], importances['importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title('Top 15 Feature Importances (RandomForest)')
    ax.grid(axis='x', alpha=0.3)
    
    print('  Model training complete')

except Exception as e:
    print(f'  Warning: Could not train model - {e}')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.5, 0.5, f'Feature importance model training failed:\n{str(e)}', 
            ha='center', va='center', fontsize=12, wrap=True)
    ax.set_title('Top 15 Feature Importances')
    metrics['top_features'] = {}

plt.tight_layout()
plt.savefig(figures_dir / 'fig12_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ fig12_feature_importance.png saved')


# =============================================================================
# Save Metrics Summary
# =============================================================================
print('\n' + '='*70)
print('ANALYSIS METRICS SUMMARY')
print('='*70)

# Print formatted metrics
print('\n1. DATA QUALITY:')
print(f'   - Total missing hours: {metrics.get("total_missing_hours", "N/A")}')
print(f'   - Average missing %: {metrics.get("avg_missing_pct", 0):.2f}%')
print(f'   - Total zero-trip hours: {metrics.get("total_zero_trips", "N/A")}')
print(f'   - Outage periods detected: {metrics.get("total_outage_periods", "N/A")}')
print(f'   - Total outage hours: {metrics.get("total_outage_hours", "N/A")}')

print('\n2. DEMAND PATTERNS:')
print(f'   - Top zone: {metrics.get("top_zone", "N/A")} ({metrics.get("top_zone_trips", 0):,} trips)')
print(f'   - Average daily trips: {metrics.get("avg_daily_trips", 0):.0f}')
print(f'   - Peak daily trips: {metrics.get("peak_daily_trips", "N/A")} on {metrics.get("peak_day", "N/A")}')
print(f'   - Peak hour: {metrics.get("peak_hour", "N/A")}:00')
print(f'   - Peak weekday: {metrics.get("peak_weekday", "N/A")}')

print('\n3. WEATHER IMPACT:')
print(f'   - Rain-trips correlation: {metrics.get("rain_trips_correlation", 0):.3f}')
print(f'   - Temp-trips correlation: {metrics.get("temp_trips_correlation", 0):.3f}')
print(f'   - Average rain lift: {metrics.get("avg_rain_lift_pct", 0):.2f}%')

print('\n4. TEMPORAL EFFECTS:')
print(f'   - Average holiday trips: {metrics.get("avg_holiday_trips", 0):.1f}')
print(f'   - Average regular trips: {metrics.get("avg_regular_trips", 0):.1f}')
print(f'   - Holiday demand change: {metrics.get("holiday_demand_change_pct", 0):.2f}%')

print('\n5. EVENT IMPACT:')
print(f'   - Football match lift: {metrics.get("football_match_lift_pct", 0):.2f}%')
if 'event_type_impacts' in metrics:
    for event_type, lift in metrics['event_type_impacts'].items():
        print(f'   - {event_type} lift: {lift:.2f}%')

print('\n' + '='*70)

# Save to JSON
metrics_file = Path('figures/analysis_metrics.json')
with open(metrics_file, 'w') as f:
    # Convert any non-serializable objects
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            metrics_serializable[k] = float(v)
        elif isinstance(v, dict):
            metrics_serializable[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating)) else v2 
                                       for k2, v2 in v.items()}
        else:
            metrics_serializable[k] = v
    
    json.dump(metrics_serializable, f, indent=2, default=str)

print(f'\n✓ Metrics saved to {metrics_file}')
print(f'✓ All 12 figures generated successfully in {figures_dir}/')
print('\nDone!')
