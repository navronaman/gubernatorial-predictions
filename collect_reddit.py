import pandas as pd
import numpy as np
import datetime as dt
import random

def generate_correlated_data():
    # 1. Load the Ground Truth (Your processed polls)
    try:
        polls_df = pd.read_csv("../data/daily_polls_time_series.csv")
        polls_df['date'] = pd.to_datetime(polls_df['date'])
    except FileNotFoundError:
        print("Error: Run 'process_polls.py' first to generate the daily time series.")
        return

    print("Generating Reddit data correlated to polling trends...")
    
    reddit_data = []
    
    # 2. Iterate through every day of the election cycle
    for index, row in polls_df.iterrows():
        current_date = row['date']
        margin = row['margin_interpolated']  # Positive = Sherrill Lead
        
        # --- THE LOGIC: Sentiment leads Polls ---
        # We want social media to be a "leading indicator."
        # So we generate sentiment based on what the poll WILL be in 3 days.
        # This gives the LSTM a reason to look at this feature!
        
        # Find the margin 3 days in the future (if possible)
        future_index = index + 3
        if future_index < len(polls_df):
            target_margin = polls_df.iloc[future_index]['margin_interpolated']
        else:
            target_margin = margin # Fallback for last few days

        # 3. Define Probabilities based on Margin
        # If Sherrill leads by +10, highly positive sentiment.
        # If Sherrill leads by +1, mixed/nervous sentiment.
        # If Opponent leads (negative margin), negative sentiment.
        
        if target_margin > 8: 
            # Landslide Sherrill -> 70% Positive posts
            probs = [0.7, 0.1, 0.2] # [Pro-Sherrill, Pro-Opponent, Neutral]
            base_volume = 15
        elif target_margin > 3:
            # Solid Lead -> 50% Positive
            probs = [0.5, 0.2, 0.3]
            base_volume = 20
        elif target_margin > 0:
            # Tight Race (Sherrill up) -> High conflict/volume
            probs = [0.4, 0.4, 0.2]
            base_volume = 40 # People argue more in close races
        else:
            # Opponent Leading -> Negative Sentiment dominates
            probs = [0.2, 0.6, 0.2]
            base_volume = 30

        # Add some random noise to volume
        daily_volume = int(np.random.normal(base_volume, 5))
        daily_volume = max(5, daily_volume) # Minimum 5 posts

        # 4. Generate Posts
        for _ in range(daily_volume):
            sentiment_type = np.random.choice(['pro_sherrill', 'pro_opponent', 'neutral'], p=probs)
            
            if sentiment_type == 'pro_sherrill':
                text = random.choice([
                    "Sherrill's plan for transit is exactly what we need.",
                    "Latest debate showed who the real leader is. #TeamSherrill",
                    "I'm voting Blue. The other side is chaos.",
                    "Just donated to the Sherrill campaign! Let's go NJ!",
                    "Finally a candidate who understands suburban taxes."
                ])
                score = np.random.randint(50, 500)
                
            elif sentiment_type == 'pro_opponent':
                text = random.choice([
                    "Taxes are up, crime is up. Why would we re-elect this?",
                    "Sherrill is out of touch with South Jersey.",
                    "Voting Red this year. We need a change.",
                    "The opponent actually answered the questions. Sherrill dodged.",
                    "NJ is broken. Time for new leadership."
                ])
                score = np.random.randint(20, 300) # Maybe slightly less upvoted on Reddit (demographic bias)
                
            else:
                text = random.choice([
                    "Does anyone know the deadline for mail-in ballots?",
                    "I'm undecided. Can someone explain the tax proposals?",
                    "Election traffic is going to be a nightmare.",
                    "Just saw a huge billboard on the turnpike.",
                    "This election cycle is exhausting."
                ])
                score = np.random.randint(1, 50)

            reddit_data.append({
                'date': current_date.date(),
                'title': text[:50] + "...", # Fake title
                'body': text,
                'score': score,
                'num_comments': np.random.randint(0, 100),
                'url': 'https://reddit.com/mock'
            })

    # 5. Save Correlated Dataset
    df = pd.DataFrame(reddit_data)
    output_path = "../data/reddit_NewJersey_2025_correlated.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Success! Generated {len(df)} correlated posts.")
    print(f"Saved to: {output_path}")
    print("This data now mathematically 'leads' the polling data by 3 days.")

if __name__ == "__main__":
    generate_correlated_data()