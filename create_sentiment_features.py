"""
Sentiment Feature Engineering Script for Phase 1 NLP

This script:
1. Loads Reddit synthetic data
2. Applies VADER sentiment analysis to text
3. Aggregates sentiment metrics by date
4. Merges with polling data to create Master Training Table

Required: vaderSentiment library
Install with: pip install vaderSentiment
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def create_sentiment_features():
    print("=" * 80)
    print("Phase 1: NLP Sentiment Feature Engineering")
    print("=" * 80)

    # Initialize VADER sentiment analyzer
    print("\n[1/6] Initializing VADER sentiment analyzer...")
    analyzer = SentimentIntensityAnalyzer()
    print("   VADER loaded successfully")

    # Load Reddit data
    print("\n[2/6] Loading Reddit synthetic data...")
    reddit_df = pd.read_csv('data/reddit_NewJersey_2025_correlated.csv')
    reddit_df['date'] = pd.to_datetime(reddit_df['date'])
    print(f"   Loaded {len(reddit_df)} Reddit posts from {reddit_df['date'].min().date()} to {reddit_df['date'].max().date()}")

    # Apply VADER sentiment analysis
    print("\n[3/6] Applying VADER sentiment analysis to all posts...")
    print("   This may take a moment...")

    def get_vader_sentiment(text):
        """Get compound sentiment score from VADER (-1 to +1)"""
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']  # Compound score ranges from -1 (negative) to +1 (positive)

    reddit_df['sentiment_score'] = reddit_df['body'].apply(get_vader_sentiment)

    print(f"   Sentiment analysis complete!")
    print(f"   Sentiment range: {reddit_df['sentiment_score'].min():.3f} to {reddit_df['sentiment_score'].max():.3f}")
    print(f"   Average sentiment: {reddit_df['sentiment_score'].mean():.3f}")

    # Aggregate by date
    print("\n[4/6] Aggregating sentiment metrics by date...")

    # Calculate daily aggregated features
    daily_sentiment = reddit_df.groupby('date').agg({
        'sentiment_score': 'mean',           # Average sentiment
        'body': 'count',                      # Post volume
        'score': 'sum',                       # Total upvotes
        'num_comments': 'sum'                 # Total comments
    }).reset_index()

    # Rename columns for clarity
    daily_sentiment.columns = ['date', 'reddit_avg_sentiment', 'reddit_post_volume',
                                'reddit_total_upvotes', 'reddit_total_comments']

    # Calculate weighted sentiment (upvotes * sentiment)
    # First calculate per-post weighted sentiment, then sum by date
    reddit_df['weighted_sentiment'] = reddit_df['score'] * reddit_df['sentiment_score']
    weighted_by_date = reddit_df.groupby('date')['weighted_sentiment'].sum().reset_index()
    weighted_by_date.columns = ['date', 'reddit_weighted_sentiment']

    # Merge weighted sentiment
    daily_sentiment = daily_sentiment.merge(weighted_by_date, on='date', how='left')

    # Normalize weighted sentiment by total upvotes to avoid scale issues
    daily_sentiment['reddit_weighted_sentiment_norm'] = (
        daily_sentiment['reddit_weighted_sentiment'] / daily_sentiment['reddit_total_upvotes']
    ).fillna(0)

    print(f"   Aggregated to {len(daily_sentiment)} daily entries")
    print(f"   Average daily post volume: {daily_sentiment['reddit_post_volume'].mean():.1f}")

    # Load polling data
    print("\n[5/6] Loading polling ground truth data...")
    polls_df = pd.read_csv('data/daily_polls_time_series.csv')
    polls_df['date'] = pd.to_datetime(polls_df['date'])
    print(f"   Loaded {len(polls_df)} days of polling data")

    # Merge polling data with sentiment features
    print("\n[6/6] Creating Master Training Table...")
    master_df = polls_df.merge(daily_sentiment, on='date', how='left')

    # Handle any NaN values (should not happen since dates align, but safety check)
    sentiment_cols = ['reddit_avg_sentiment', 'reddit_post_volume', 'reddit_total_upvotes',
                     'reddit_total_comments', 'reddit_weighted_sentiment',
                     'reddit_weighted_sentiment_norm']

    # Fill NaN with 0 (neutral sentiment if no data)
    for col in sentiment_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(0)

    # Save Master Training Table
    output_path = 'data/master_training_table.csv'
    master_df.to_csv(output_path, index=False)

    print(f"   Master Training Table saved to: {output_path}")
    print(f"   Total rows: {len(master_df)}")
    print(f"   Total columns: {len(master_df.columns)}")

    # Display summary
    print("\n" + "=" * 80)
    print("Master Training Table - Column Summary")
    print("=" * 80)
    print("\nTarget Variable:")
    print("  - margin_interpolated: Sherrill lead margin (Sherrill % - Opponent %)")

    print("\nPolling Features:")
    print("  - sherrill_pct: Sherrill polling percentage")
    print("  - opponent_pct: Opponent polling percentage")
    print("  - days_until_election: Days remaining until election")

    print("\nReddit Sentiment Features:")
    print(f"  - reddit_avg_sentiment: Average daily sentiment ({master_df['reddit_avg_sentiment'].mean():.3f} avg)")
    print(f"  - reddit_post_volume: Number of posts per day ({master_df['reddit_post_volume'].mean():.1f} avg)")
    print(f"  - reddit_total_upvotes: Total upvotes per day ({master_df['reddit_total_upvotes'].mean():.1f} avg)")
    print(f"  - reddit_total_comments: Total comments per day ({master_df['reddit_total_comments'].mean():.1f} avg)")
    print(f"  - reddit_weighted_sentiment: Upvote-weighted sentiment")
    print(f"  - reddit_weighted_sentiment_norm: Normalized weighted sentiment")

    # Show correlation with target variable
    print("\n" + "=" * 80)
    print("Feature Correlation with Target (margin_interpolated)")
    print("=" * 80)
    correlations = master_df.corr()['margin_interpolated'].sort_values(ascending=False)
    print(correlations.to_string())

    # Show first few rows
    print("\n" + "=" * 80)
    print("First 5 Rows of Master Training Table")
    print("=" * 80)
    print(master_df.head().to_string(index=False))

    # Show last few rows
    print("\n" + "=" * 80)
    print("Last 5 Rows of Master Training Table")
    print("=" * 80)
    print(master_df.tail().to_string(index=False))

    print("\n" + "=" * 80)
    print("SUCCESS! Sentiment feature engineering complete.")
    print("Master Training Table is ready for LSTM modeling.")
    print("=" * 80)

    return master_df

if __name__ == "__main__":
    # Check if vaderSentiment is installed
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        create_sentiment_features()
    except ImportError:
        print("ERROR: vaderSentiment not installed.")
        print("Please install it with: pip install vaderSentiment")
        print("Or: uv pip install vaderSentiment")
