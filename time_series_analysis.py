"""
Step 2 - Exploratory Time Series Analysis and Visualization
This script performs exploratory data analysis and visualization on the daily interpolated polling data for the NJ Governor 2025
Main purpose of this step is to create a uniform time series dataset, the visualization is secondary but important for understanding trends.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

sns.set_style("whitegrid")

plt.rcParams['figure.figsize'] = (14, 8)

df = pd.read_csv('nj2025_270toWin.csv')

df['start_date'] = pd.to_datetime(df['start_date'])

df['end_date'] = pd.to_datetime(df['end_date'])

df['poll_date'] = df['start_date'] + (df['end_date'] - df['start_date']) / 2

df = df.sort_values('poll_date')

df['spread'] = df['sherrill_pct'] - df['opponent_pct']

print(df[['poll_date', 'pollster', 'sherrill_pct', 'opponent_pct', 'spread', 'sample_size']].head())

df.to_csv('time_series_data.csv', index=False)

# Here ends the substantive portion of the codefile
# Below is the visualization code, which again, is not part of the submission per se

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
ax1.plot(df['poll_date'], df['sherrill_pct'], 'o-', label='Sherrill', color='blue', linewidth=2, markersize=8)
ax1.plot(df['poll_date'], df['opponent_pct'], 's-', label='Opponent', color='red', linewidth=2, markersize=8)

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

ax2 = axes[1]
colors = ['green' if x > 0 else 'orange' for x in df['spread']]
ax2.bar(df['poll_date'], df['spread'], color=colors, alpha=0.6, width=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

z3 = np.polyfit(df.index, df['spread'], 1)
p3 = np.poly1d(z3)
ax2.plot(df['poll_date'], p3(df.index), "--", color='darkblue', linewidth=2, label='Spread Trend')

ax2.set_xlabel('Poll Date', fontsize=12)
ax2.set_ylabel('Spread (Sherrill - Opponent)', fontsize=12)
ax2.set_title('Polling Spread Over Time', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

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
print("Time series visualization saved to 'time_series_analysis.png'")

fig2, ax = plt.subplots(figsize=(14, 8))

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