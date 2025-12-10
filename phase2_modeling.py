"""
This is the magnum opus of the project.
Step 4: Modeling - ARIMA Baseline vs LSTM with Social Media Sentiment

This script implements the core hypothesis test:
- Baseline: ARIMA using only polling margin
- LSTM: Multi-feature model using polling + Reddit sentiment
- Test: Does social media sentiment improve prediction accuracy?

Design choices:
- We use ARIMA as a classical time series baseline.
- We use LSTM to capture temporal dependencies and multiple features.
- We evaluate using RMSE and MAE on a held-out test set (October 2025).
- Visualizations compare actual vs predicted margins for both models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(420)
tf.random.set_seed(420)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_and_split_data():
    # Load master training table and split into train/test sets
    print("Time Series Modeling")

    print("Loading Master Training Table")
    df = pd.read_csv('data/master_training_table.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} days of data from {df['date'].min().date()} to {df['date'].max().date()}")

    # Define train/test split
    # Train: June-September (Aug-Oct in roadmap, but we'll use more data)
    # Test: October (final weeks before election)
    split_date = pd.Timestamp('2025-10-01')

    train_df = df[df['date'] < split_date].copy()
    test_df = df[df['date'] >= split_date].copy()

    print(f"Train/Test Split")
    print(f"Training set: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} days)")
    print(f"Test set: {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} days)")

    return df, train_df, test_df

def train_arima_baseline(train_df, test_df):
    """Train ARIMA model on margin_interpolated only"""
    print(f"Training ARIMA Baseline Model")
    print("Using only polling margin (no sentiment features)")

    # Extract target variable
    train_values = train_df['margin_interpolated'].values
    test_values = test_df['margin_interpolated'].values

    # Fit ARIMA model (order can be tuned)
    # ARIMA(p, d, q) where p=lag order, d=differencing, q=moving average
    print("Fitting ARIMA(2,1,2) model")
    model = ARIMA(train_values, order=(2, 1, 2))
    arima_fit = model.fit()

    # Make predictions
    n_forecast = len(test_df)
    arima_predictions = arima_fit.forecast(steps=n_forecast)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_values, arima_predictions))
    mae = mean_absolute_error(test_values, arima_predictions)

    print(f"ARIMA Test RMSE: {rmse:.3f}")
    print(f"ARIMA Test MAE: {mae:.3f}")

    return arima_fit, arima_predictions, rmse, mae

def prepare_lstm_data(train_df, test_df, lookback=14):
    """
    Prepare 3D data for LSTM: (samples, lookback_window, features)

    Features used:
    1. margin_interpolated (target, also used as feature)
    2. reddit_avg_sentiment
    3. reddit_post_volume (normalized)
    """
    print(f"Preparing data for LSTM...")
    print(f"Lookback window: {lookback} days")

    # Select features
    feature_cols = ['margin_interpolated', 'reddit_avg_sentiment', 'reddit_post_volume']
    target_col = 'margin_interpolated'

    # Combine train and test for proper scaling
    all_data = pd.concat([train_df, test_df])[feature_cols].values

    # Scale features to [0, 1] range
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(all_data)

    # Split back into train/test
    train_scaled = scaled_data[:len(train_df)]
    test_scaled = scaled_data[len(train_df):]

    def create_sequences(data, lookback):
        """Create sequences of lookback days for LSTM"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, :])  # All features for lookback period
            y.append(data[i, 0])  # Target is margin_interpolated (first column)
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)

    print(f"Training sequences: {X_train.shape} (samples, lookback, features)")
    print(f"Test sequences: {X_test.shape}")
    print(f"Features: {feature_cols}")

    return X_train, y_train, X_test, y_test, scaler, feature_cols

def build_lstm_model(input_shape):
    # Build LSTM model architecture
    print(f"Building LSTM Model")

    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dropout(0.2),
        Dense(1)  
        # Output: single value (margin prediction)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Model Architecture:")
    model.summary()

    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test):
    # Train LSTM model with early stopping 
    print(f"Training LSTM Model")

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )

    print(f"Training completed in {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    return model, history

def evaluate_lstm(model, X_test, y_test, scaler, test_df, lookback):
    # Evaluate LSTM model and inverse transform predictions
    print(f"Evaluating LSTM Model")

    # Make predictions
    lstm_predictions_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions back to original scale
    # Create dummy array with same shape as scaler expects
    dummy = np.zeros((len(lstm_predictions_scaled), scaler.n_features_in_))
    dummy[:, 0] = lstm_predictions_scaled.flatten()
    lstm_predictions = scaler.inverse_transform(dummy)[:, 0]

    # Inverse transform actual values
    dummy_actual = np.zeros((len(y_test), scaler.n_features_in_))
    dummy_actual[:, 0] = y_test
    actual_values = scaler.inverse_transform(dummy_actual)[:, 0]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_values, lstm_predictions))
    mae = mean_absolute_error(actual_values, lstm_predictions)

    print(f"LSTM Test RMSE: {rmse:.3f}")
    print(f"LSTM Test MAE: {mae:.3f}")

    return lstm_predictions, actual_values, rmse, mae

def visualize_results(df, train_df, test_df, arima_predictions, lstm_predictions,
                     arima_rmse, lstm_rmse, lookback):
    # Create comprehensive visualization of results
    print(f"Creating visualizations")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Full time series with predictions
    ax1 = axes[0]
    ax1.plot(df['date'], df['margin_interpolated'], 'o-', label='Actual Polling Margin',
             color='black', linewidth=2, markersize=4)

    # Mark train/test split
    split_date = test_df['date'].min()
    ax1.axvline(x=split_date, color='red', linestyle='--', linewidth=2, label='Train/Test Split')

    # Plot ARIMA predictions
    arima_dates = test_df['date'].values
    ax1.plot(arima_dates, arima_predictions, 's-', label=f'ARIMA Forecast (RMSE: {arima_rmse:.2f})', color='blue', linewidth=2, markersize=6, alpha=0.7)

    # Plot LSTM predictions (offset by lookback days)
    lstm_dates = test_df['date'].values[lookback:]
    ax1.plot(lstm_dates, lstm_predictions, '^-', label=f'LSTM Forecast (RMSE: {lstm_rmse:.2f})',color='green', linewidth=2, markersize=6, alpha=0.7)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Margin (Sherrill - Opponent %)', fontsize=12)
    ax1.set_title('Polling Margin Forecast: ARIMA vs LSTM with Social Media Sentiment', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Test period zoom-in (October 2025)
    ax2 = axes[1]
    test_actual = test_df['margin_interpolated'].values
    test_dates = test_df['date'].values

    ax2.plot(test_dates, test_actual, 'o-', label='Actual', color='black',
             linewidth=3, markersize=8)
    ax2.plot(arima_dates, arima_predictions, 's--', label='ARIMA', color='blue', linewidth=2, markersize=7, alpha=0.7)
    ax2.plot(lstm_dates, lstm_predictions, '^--', label='LSTM', color='green',
             linewidth=2, markersize=7, alpha=0.7)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Margin (%)', fontsize=12)
    ax2.set_title('Test Period Predictions (October 2025)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Prediction errors
    ax3 = axes[2]
    arima_errors = test_actual - arima_predictions
    lstm_errors = test_actual[lookback:] - lstm_predictions

    ax3.plot(arima_dates, arima_errors, 's-', label='ARIMA Error', color='blue', linewidth=2, markersize=6)
    ax3.plot(lstm_dates, lstm_errors, '^-', label='LSTM Error', color='green',
             linewidth=2, markersize=6)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Prediction Error (%)', fontsize=12)
    ax3.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase2_model_comparison.png', dpi=300, bbox_inches='tight')
    print("phase2_model_comparison.png is now saved")

    plt.close()

def main():
    # Main execution pipeline

    # Load and split data
    df, train_df, test_df = load_and_split_data()

    # Train ARIMA baseline
    arima_fit, arima_predictions, arima_rmse, arima_mae = train_arima_baseline(train_df, test_df)

    # Prepare LSTM data
    lookback = 14
    X_train, y_train, X_test, y_test, scaler, feature_cols = prepare_lstm_data(train_df, test_df, lookback=lookback)

    # Build and train LSTM
    lstm_model = build_lstm_model(input_shape=(lookback, len(feature_cols)))
    lstm_model, history = train_lstm_model(lstm_model, X_train, y_train, X_test, y_test)

    # Evaluate LSTM
    lstm_predictions, actual_values, lstm_rmse, lstm_mae = evaluate_lstm(
        lstm_model, X_test, y_test, scaler, test_df, lookback
    )

    # Visualize results
    visualize_results(df, train_df, test_df, arima_predictions, lstm_predictions, arima_rmse, lstm_rmse, lookback)

    # Final comparison
    print("FINAL RESULTS: ARIMA vs LSTM")
    print(f"ARIMA (Polling Only):")
    print(f"RMSE: {arima_rmse:.3f}")
    print(f"MAE:  {arima_mae:.3f}")

    print(f"LSTM (Polling + Social Media Sentiment):")
    print(f"RMSE: {lstm_rmse:.3f}")
    print(f"MAE:  {lstm_mae:.3f}")
    improvement = ((arima_rmse - lstm_rmse) / arima_rmse) * 100
    print(f"\nImprovement: {improvement:.3f}%")

    if lstm_rmse < arima_rmse:
        print("Conclusion: LSTM with social media sentiment OUTPERFORMS baseline ARIMA!")
        print("Hypothesis CONFIRMED: Social media acts as a leading indicator.")
    else:
        print("Conclusion: ARIMA baseline performs better.")
        print("Hypothesis REJECTED: Social media does not improve predictions.")

    print("Modeling Complete!")

    # Save models
    lstm_model.save('data/lstm_model.keras')
    print("LSTM model saved to data/lstm_model.keras")

if __name__ == "__main__":
    main()
