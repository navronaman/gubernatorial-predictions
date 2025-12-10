"""
Step 1: Daily Interpolation Script for NJ Governor 2025 Polling Data

This script converts sparse polling data into a daily time series by:
1. Reading the raw polling data with 23 polls
2. Interpolating polling percentages to daily frequency
3. Calculating daily margin (Sherrill - Citarelli)
4. Adding days_until_election feature
    We will be using this as a temporal feature for modeling
5. Saving to data/daily_polls_time_series.csv

Some other design choices:
We have decided to use a functional method, where each python script is a function that can be imported and called from other scripts.
This allows for modularity and reusability across different parts of the project.

Some assumptions:
- Election Day is assumed to be November 4, 2025
- Linear interpolation is used for missing values
- The output CSV will have columns: date, sherrill_pct, opponent_pct, margin_interpolated, days_until_election
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_daily_interpolation():
    print("Daily Polling Interpolation Script")
    # Load the sparse polling data
    print("Loading sparse polling data")
    df = pd.read_csv('time_series_data.csv')
    df['poll_date'] = pd.to_datetime(df['poll_date'])
    df = df.sort_values('poll_date').reset_index(drop=True)

    print(f"Loaded {len(df)} polls from {df['poll_date'].min().date()} to {df['poll_date'].max().date()}")
    # There are about 23 polls in total, runninig from September to November
    # Create a complete daily date range
    print("Creating daily date range")
    start_date = df['poll_date'].min().date()
    end_date = df['poll_date'].max().date()

    # Assuming election day is approximately Nov 4, 2025
    election_day = end_date + timedelta(days=8)  # Oct 27 + 8 days = Nov 4
    print(f"Assumed Election Day: {election_day}")

    # Create daily date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_df = pd.DataFrame({'date': date_range})

    print(f"Created {len(daily_df)} daily entries")
    # Prepare polling data for interpolation
    print("Interpolating polling percentages")

    # Create a dataframe with just poll dates and values
    polls_only = df[['poll_date', 'sherrill_pct', 'opponent_pct', 'spread']].copy()
    polls_only.columns = ['date', 'sherrill_pct', 'opponent_pct', 'spread']
    polls_only['date'] = polls_only['date'].dt.date

    # Merge with daily range (left join to keep all daily dates)
    daily_df['date_key'] = daily_df['date'].dt.date
    daily_df = daily_df.merge(polls_only, left_on='date_key', right_on='date', how='left', suffixes=('', '_poll'))
    daily_df = daily_df.drop(columns=['date_poll', 'date_key'])

    # Interpolate missing values using linear interpolation
    daily_df['sherrill_pct'] = daily_df['sherrill_pct'].interpolate(method='linear')
    daily_df['opponent_pct'] = daily_df['opponent_pct'].interpolate(method='linear')

    # Calculate interpolated margin
    daily_df['margin_interpolated'] = daily_df['sherrill_pct'] - daily_df['opponent_pct']

    # Add days_until_election feature
    print("Adding temporal features")
    daily_df['days_until_election'] = (pd.to_datetime(election_day) - daily_df['date']).dt.days

    # Round to 2 decimal places for cleanliness
    daily_df['sherrill_pct'] = daily_df['sherrill_pct'].round(2)
    daily_df['opponent_pct'] = daily_df['opponent_pct'].round(2)
    daily_df['margin_interpolated'] = daily_df['margin_interpolated'].round(2)

    # Format date as string (YYYY-MM-DD)
    daily_df['date'] = daily_df['date'].dt.strftime('%Y-%m-%d')

    # Reorder columns
    daily_df = daily_df[['date', 'sherrill_pct', 'opponent_pct', 'margin_interpolated', 'days_until_election']]

    # Save to data/ directory
    print("Saving daily interpolated data")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    output_path = 'data/daily_polls_time_series.csv'
    daily_df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(daily_df)}")

    # Display summary statistics
    print("Summary Statistics")
    print(f"Date Range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Total Days: {len(daily_df)}")
    print(f"Sherrill Average: {daily_df['sherrill_pct'].mean():.2f}%")
    print(f"Opponent Average: {daily_df['opponent_pct'].mean():.2f}%")
    print(f"Average Margin: {daily_df['margin_interpolated'].mean():.2f}%")
    print(f"Margin Range: {daily_df['margin_interpolated'].min():.2f}% to {daily_df['margin_interpolated'].max():.2f}%")

    print("First 5 Days:")
    print(daily_df.head().to_string(index=False))

    print("Last 5 Days:")
    print(daily_df.tail().to_string(index=False))

    print("Daily interpolation complete.")

    return daily_df

if __name__ == "__main__":
    create_daily_interpolation()
