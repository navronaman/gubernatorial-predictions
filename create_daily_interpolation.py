"""
Step 3: Daily Interpolation Script for NJ Governor 2025 Polling Data

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
    # Load the sparse polling data
    df = pd.read_csv('time_series_data.csv')
    df['poll_date'] = pd.to_datetime(df['poll_date'])
    df = df.sort_values('poll_date').reset_index(drop=True)

    # There are about 23 polls in total, runninig from September to November
    # Create a complete daily date range
    start_date = df['poll_date'].min().date()
    end_date = df['poll_date'].max().date()

    # Assuming election day is approximately Nov 4, 2025
    election_day = end_date + timedelta(days=8)  # Oct 27 + 8 days = Nov 4
    print(election_day)

    # Create daily date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_df = pd.DataFrame({'date': date_range})

    # Prepare polling data for interpolation
    # Create a dataframe with just poll dates and values
    polls_only = df[['poll_date', 'sherrill_pct', 'opponent_pct', 'spread']].copy()
    polls_only.columns = ['date', 'sherrill_pct', 'opponent_pct', 'spread']
    polls_only['date'] = polls_only['date'].dt.date

    # Merge with daily range
    # We have used left join to keep all daily dates
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

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    output_path = 'data/daily_polls_time_series.csv'
    daily_df.to_csv(output_path, index=False)

    return daily_df

if __name__ == "__main__":
    create_daily_interpolation()
