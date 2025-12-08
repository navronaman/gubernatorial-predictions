# Project Roadmap: "Pollsters" (NJ Governor 2025 Prediction)

## Context for LLM
This project is a **multimodal time-series forecasting initiative**. The goal is to predict daily polling numbers for the 2025 NJ Gubernatorial Election (Sherrill vs. Opponent) by fusing traditional polling data with synthetic social media sentiment (Reddit/Twitter).

---

## 1. Project Objective

**Core Question:** Can social media sentiment (which is fast and volatile) act as a "leading indicator" to predict future polling trends (which are slow and sparse) using Deep Learning (LSTM)?

**Hypothesis:** An LSTM model trained on a sequence of `[Poll_Margin, Reddit_Sentiment, Twitter_Sentiment]` will outperform a baseline ARIMA model that uses `[Poll_Margin]` alone.

---

## 2. Current State (Data Layer)

We have successfully bypassed API limitations by generating robust synthetic data correlated with real ground-truth polls.

### A. Ground Truth (Real Data)
- **File:** `data/daily_polls_time_series.csv`
- **Source:** Manually transcribed from FiveThirtyEight/RCP, then interpolated to daily frequency
- **Key Columns:**
  - `date`: YYYY-MM-DD
  - `margin_interpolated`: Float (Target Variable). Positive = Sherrill Lead
  - `days_until_election`: Integer

### B. Feature Set 1 (Reddit - Synthetic Correlated)
- **File:** `data/reddit_NewJersey_2025_correlated.csv`
- **Generation Logic:** Generated to mathematically "lead" polling trends by 3 days (simulating early detection)
- **Key Columns:**
  - `date`: YYYY-MM-DD
  - `body`: String (Text content of the post)
  - `score`: Integer (Upvotes)

### C. Feature Set 2 (Twitter - Synthetic Volatile)
- **File:** `data/twitter_NewJersey_2025.csv`
- **Generation Logic:** Higher volume, more noise, higher volatility
- **Key Columns:**
  - `date`: YYYY-MM-DD
  - `text`: String (Tweet content)
  - `retweets`: Integer

---

## 3. Development Roadmap (Next Steps)

### Phase 1: Feature Engineering (NLP) -> **[CURRENT PRIORITY]**

The LLM needs to write scripts to convert raw text into numerical time-series features.

1. **Sentiment Scoring:** Apply VADER (Valence Aware Dictionary and sEntiment Reasoner) to `body` and `text` columns
2. **Daily Aggregation:** Group by date to calculate:
   - `avg_sentiment_score`
   - `post_volume`
   - `weighted_sentiment` (Score * Sentiment)
3. **Data Merging:** Left join these aggregated features onto `daily_polls_time_series.csv` to create the **Master Training Table**

### Phase 2: Modeling (Deep Learning)

1. **Baseline:** Train an ARIMA model on `margin_interpolated` alone. Record RMSE
2. **Data Prep:** Convert the Master Table into 3D Numpy arrays for LSTM: `(Samples, Lookback_Window=14, Features=3)`
3. **LSTM Construction:** Build a Keras/TensorFlow model:
   - Input Layer -> LSTM (50 units) -> Dropout -> Dense (1 unit)
4. **Training:** Train on Aug-Oct data, Test on final Nov weeks

### Phase 3: Evaluation & Dashboard

1. **Comparison:** Plot Actual vs. ARIMA vs. LSTM predictions
2. **Visualization:** Build a Streamlit app to display the "Leading Indicator" effect (show how the model catches the trend early)

---

## 4. Instructions for LLM Code Generation

- **Library Constraints:** Use `pandas`, `numpy`, `vaderSentiment`, `scikit-learn`, `tensorflow/keras`
- **Data Handling:** Always handle NaN values resulting from merges (fill with 0 or forward fill)
- **Dates:** Ensure all merges are performed on datetime objects, not strings
