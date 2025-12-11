"""
Step 5: Sentiment Feature Engineering Script for Phase 1 NLP

This script:
1. Loads Reddit synthetic data
2. Applies VADER sentiment analysis to text
3. Aggregates sentiment metrics by date
4. Merges with polling data to create Master Training Table

Required: vaderSentiment library
Install with: pip install vaderSentiment

Some design choices:
- We have decided to use a functional method, where each python script is a function that can be imported and called from other scripts.
This allows for modularity and reusability across different parts of the project.

Why VADER?
- VADER is specifically tuned for social media text, making it ideal for Reddit data.
- It provides a compound sentiment score ranging from -1 (most negative) to +1 (most positive), which is useful for quantitative analysis.

"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def create_sentiment_features():
    print("NLP Sentiment Feature Engineering")

    # Initialize VADER sentiment analyzer
    print("Initializing VADER sentiment analyzer")
    analyzer = SentimentIntensityAnalyzer()
    print("VADER loaded successfully")

    # Load Reddit data
    print("Loading Reddit synthetic data")
    reddit_df = pd.read_csv('data/reddit_NewJersey_2025_correlated.csv')
    reddit_df['date'] = pd.to_datetime(reddit_df['date'])
    print(f"Loaded {len(reddit_df)} Reddit posts from {reddit_df['date'].min().date()} to {reddit_df['date'].max().date()}")

    # Apply VADER sentiment analysis
    print("Applying VADER sentiment analysis to all posts")
    print("This may take a moment")
    def get_vader_sentiment(text):
        """Get compound sentiment score from VADER (-1 to +1)"""
        scores = analyzer.polarity_scores(str(text))
        return scores['compound']  
        # Compound score ranges from -1 (negative) to +1 (positive)

    reddit_df['sentiment_score'] = reddit_df['body'].apply(get_vader_sentiment)

    print(f"Sentiment analysis complete!")
    print(f"Sentiment range: {reddit_df['sentiment_score'].min():.3f} to {reddit_df['sentiment_score'].max():.3f}")
    print(f"Average sentiment: {reddit_df['sentiment_score'].mean():.3f}")

    # Aggregate by date
    print("Aggregating sentiment metrics by date")

    # Calculate daily aggregated features
    daily_sentiment = reddit_df.groupby('date').agg({
        'sentiment_score': 'mean',           
        # Average sentiment
        'body': 'count',                      
        # Post volume
        'score': 'sum',                       
        # Total upvotes
        'num_comments': 'sum'                 
        # Total comments
    }).reset_index()

    # Rename columns for clarity
    daily_sentiment.columns = ['date', 'reddit_avg_sentiment', 'reddit_post_volume',
                                'reddit_total_upvotes', 'reddit_total_comments']

    # Calculate weighted sentiment (upvotes time sentiment)
    # First calculate per-post weighted sentiment, then we will sum by date
    reddit_df['weighted_sentiment'] = reddit_df['score'] * reddit_df['sentiment_score']
    weighted_by_date = reddit_df.groupby('date')['weighted_sentiment'].sum().reset_index()
    weighted_by_date.columns = ['date', 'reddit_weighted_sentiment']

    # Merge weighted sentiment
    daily_sentiment = daily_sentiment.merge(weighted_by_date, on='date', how='left')
    # Normalize weighted sentiment by total upvotes to avoid scale issues
    daily_sentiment['reddit_weighted_sentiment_norm'] = (
        daily_sentiment['reddit_weighted_sentiment'] / daily_sentiment['reddit_total_upvotes']
    ).fillna(0)


    # Load polling data
    print("Loading polling ground truth data")
    polls_df = pd.read_csv('data/daily_polls_time_series.csv')
    polls_df['date'] = pd.to_datetime(polls_df['date'])
    print(f"Loaded {len(polls_df)} days of polling data")

    # Merge polling data with sentiment features
    print("Creating Master Training Table")
    master_df = polls_df.merge(daily_sentiment, on='date', how='left')

    # Handle any NaN values (should not happen since dates align, but safety check)
    sentiment_cols = ['reddit_avg_sentiment', 'reddit_post_volume', 'reddit_total_upvotes',
                     'reddit_total_comments', 'reddit_weighted_sentiment',
                     'reddit_weighted_sentiment_norm']

    # Fill NaN with 0 (neutral sentiment if no data)
    # This is a good practice 
    for col in sentiment_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].fillna(0)

    # Save Master Training Table
    output_path = 'data/master_training_table.csv'
    master_df.to_csv(output_path, index=False)

    return master_df

if __name__ == "__main__":
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    create_sentiment_features()
