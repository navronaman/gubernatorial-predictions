import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Read the raw data
df = pd.read_csv('raw_data.csv')

# Convert date columns to datetime
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Create a midpoint date for the poll (when it was conducted)
df['poll_date'] = df['start_date'] + (df['end_date'] - df['start_date']) / 2

# Sort by poll_date
df = df.sort_values('poll_date')

# Calculate the spread
df['spread'] = df['sherrill_pct'] - df['opponent_pct']

# Display the time series dataset
print("Time Series Dataset:")
print("=" * 80)
print(df[['poll_date', 'pollster', 'sherrill_pct', 'opponent_pct', 'spread', 'sample_size']].to_string(index=False))
print("\n")

# Save the time series dataset
df.to_csv('time_series_data.csv', index=False)
print("Time series dataset saved to 'time_series_data.csv'\n")

# Create figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Polling percentages over time
ax1 = axes[0]
ax1.plot(df['poll_date'], df['sherrill_pct'], 'o-', label='Sherrill', color='blue', linewidth=2, markersize=8)
ax1.plot(df['poll_date'], df['opponent_pct'], 's-', label='Opponent', color='red', linewidth=2, markersize=8)

# Add trend lines
z1 = np.polyfit(df.index, df['sherrill_pct'], 1)
p1 = np.poly1d(z1)
z2 = np.polyfit(df.index, df['opponent_pct'], 1)
p2 = np.poly1d(z2)
ax1.plot(df['poll_date'], p1(df.index), "--", alpha=0.7, color='blue', label='Sherrill Trend')
ax1.plot(df['poll_date'], p2(df.index), "--", alpha=0.7, color='red', label='Opponent Trend')

ax1.set_xlabel('Poll Date', fontsize=12)
ax1.set_ylabel('Polling Percentage (%)', fontsize=12)
ax1.set_title('Polling Percentages Over Time', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(30, 60)

# Plot 2: Spread over time
ax2 = axes[1]
colors = ['green' if x > 0 else 'orange' for x in df['spread']]
ax2.bar(df['poll_date'], df['spread'], color=colors, alpha=0.6, width=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add trend line for spread
z3 = np.polyfit(df.index, df['spread'], 1)
p3 = np.poly1d(z3)
ax2.plot(df['poll_date'], p3(df.index), "--", color='darkblue', linewidth=2, label='Spread Trend')

ax2.set_xlabel('Poll Date', fontsize=12)
ax2.set_ylabel('Spread (Sherrill - Opponent)', fontsize=12)
ax2.set_title('Polling Spread Over Time', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: 7-day rolling average (approximated by 3-poll rolling average)
ax3 = axes[2]
df['sherrill_ma'] = df['sherrill_pct'].rolling(window=3, min_periods=1).mean()
df['opponent_ma'] = df['opponent_pct'].rolling(window=3, min_periods=1).mean()

ax3.plot(df['poll_date'], df['sherrill_ma'], 'o-', label='Sherrill (3-poll MA)', color='darkblue', linewidth=2.5, markersize=6)
ax3.plot(df['poll_date'], df['opponent_ma'], 's-', label='Opponent (3-poll MA)', color='darkred', linewidth=2.5, markersize=6)
ax3.fill_between(df['poll_date'], df['sherrill_ma'], df['opponent_ma'],
                  where=(df['sherrill_ma'] >= df['opponent_ma']), alpha=0.2, color='blue', label='Sherrill Lead')
ax3.fill_between(df['poll_date'], df['sherrill_ma'], df['opponent_ma'],
                  where=(df['sherrill_ma'] < df['opponent_ma']), alpha=0.2, color='red', label='Opponent Lead')

ax3.set_xlabel('Poll Date', fontsize=12)
ax3.set_ylabel('Polling Percentage (%)', fontsize=12)
ax3.set_title('3-Poll Moving Average', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(30, 60)

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
print("Time series visualization saved to 'time_series_analysis.png'\n")

# Create additional detailed scatter plot
fig2, ax = plt.subplots(figsize=(14, 8))

# Create scatter plot with different markers for each pollster
pollsters = df['pollster'].unique()
colors_palette = plt.cm.tab20(np.linspace(0, 1, len(pollsters)))

for i, pollster in enumerate(pollsters):
    pollster_data = df[df['pollster'] == pollster]
    ax.scatter(pollster_data['poll_date'], pollster_data['sherrill_pct'],
               label=pollster, color=colors_palette[i], s=100, alpha=0.7, edgecolors='black')

ax.set_xlabel('Poll Date', fontsize=12)
ax.set_ylabel('Sherrill Polling Percentage (%)', fontsize=12)
ax.set_title('Sherrill Polling by Pollster Over Time', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pollster_comparison.png', dpi=300, bbox_inches='tight')
print("Pollster comparison saved to 'pollster_comparison.png'\n")

# Print summary statistics
print("\nSummary Statistics:")
print("=" * 80)
print(f"Date Range: {df['poll_date'].min().strftime('%Y-%m-%d')} to {df['poll_date'].max().strftime('%Y-%m-%d')}")
print(f"\nSherrill Average: {df['sherrill_pct'].mean():.1f}%")
print(f"Sherrill Std Dev: {df['sherrill_pct'].std():.1f}%")
print(f"Sherrill Range: {df['sherrill_pct'].min():.0f}% - {df['sherrill_pct'].max():.0f}%")
print(f"\nOpponent Average: {df['opponent_pct'].mean():.1f}%")
print(f"Opponent Std Dev: {df['opponent_pct'].std():.1f}%")
print(f"Opponent Range: {df['opponent_pct'].min():.0f}% - {df['opponent_pct'].max():.0f}%")
print(f"\nAverage Spread: {df['spread'].mean():.1f}%")
print(f"Total Polls: {len(df)}")
print(f"Number of Pollsters: {df['pollster'].nunique()}")

print("\nAnalysis complete! Check the generated PNG files for visualizations.")
