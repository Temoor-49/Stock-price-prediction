import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import os
from datetime import datetime, timedelta
from preprocess_lstm import load_data

def predict_next_days(n_days=30, sequence_length=60):
    """
    Predict stock prices for the next n days using LSTM model.
    """
    # Get project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load model and scaler
    print("Loading LSTM model and scaler...")
    model_path = os.path.join(project_dir, "models", "lstm_model.keras")
    scaler_path = os.path.join(project_dir, "models", "lstm_scaler.joblib")
    
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load historical data
    print("Loading historical data...")
    data_path = os.path.join(project_dir, "data", "aapl_stock.csv")
    df = load_data(data_path)
    
    # Get the last sequence_length days
    recent_data = df['Close'].values[-sequence_length:]
    last_price = recent_data[-1]
    
    # Scale the data
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    
    # Prepare for prediction
    predictions = []
    current_sequence = recent_data_scaled.copy()
    
    print(f"\nPredicting next {n_days} days...")
    print(f"Last known price: ${last_price:.2f}")
    print("-" * 60)
    
    # Predict iteratively
    for day in range(n_days):
        # Reshape for LSTM input: (1, sequence_length, 1)
        X_input = current_sequence.reshape(1, sequence_length, 1)
        
        # Predict next day
        pred_scaled = model.predict(X_input, verbose=0)
        
        # Inverse transform to get actual price
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_price)
        
        # Update sequence: remove first value, add prediction
        current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        if (day + 1) % 5 == 0 or day == 0:
            print(f"Day {day + 1:2d}: ${pred_price:.2f}")
    
    print("-" * 60)
    print("✓ Predictions complete!\n")
    
    # Create dates for predictions
    today = datetime.now().date()
    future_dates = [today + timedelta(days=i+1) for i in range(n_days)]
    
    # Create results DataFrame
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })
    
    return predictions_df, df, last_price

def visualize_predictions(predictions_df, historical_df, last_price, days_to_show=90):
    """Visualize historical data and future predictions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Historical + Predictions
    historical_prices = historical_df['Close'].values[-days_to_show:]
    historical_indices = range(len(historical_prices))
    
    future_indices = range(len(historical_prices), 
                          len(historical_prices) + len(predictions_df))
    
    axes[0, 0].plot(historical_indices, historical_prices, 
                   label='Historical Prices', linewidth=2, color='blue')
    axes[0, 0].plot(future_indices, predictions_df['Predicted_Price'].values,
                   label='Predicted Prices', linewidth=2, color='red', linestyle='--')
    axes[0, 0].axvline(x=len(historical_prices)-1, color='green', 
                      linestyle=':', linewidth=2, label='Today')
    axes[0, 0].set_title('Stock Price: Historical and Future Predictions', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions with Confidence Band
    pred_prices = predictions_df['Predicted_Price'].values
    days = range(len(pred_prices))
    
    # Simple confidence interval (±5%)
    confidence = 0.05
    upper_bound = pred_prices * (1 + confidence)
    lower_bound = pred_prices * (1 - confidence)
    
    axes[0, 1].plot(days, pred_prices, linewidth=2, color='red', 
                   marker='o', markersize=4, label='Predicted Price')
    axes[0, 1].fill_between(days, lower_bound, upper_bound, 
                           alpha=0.3, color='red', label='Confidence Band (±5%)')
    axes[0, 1].set_title('Future Predictions with Confidence Band', 
                        fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Days from Now')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Daily Price Changes
    daily_changes = np.diff(pred_prices, prepend=last_price)
    daily_change_pct = (daily_changes / np.append([last_price], pred_prices[:-1])) * 100
    
    colors = ['green' if x > 0 else 'red' for x in daily_changes]
    axes[1, 0].bar(days, daily_changes, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_title('Predicted Daily Price Changes', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Days from Now')
    axes[1, 0].set_ylabel('Price Change ($)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative Return
    cumulative_return = ((pred_prices - last_price) / last_price) * 100
    
    axes[1, 1].plot(days, cumulative_return, linewidth=2, color='purple', marker='o', markersize=4)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].fill_between(days, 0, cumulative_return, 
                           where=(cumulative_return >= 0), 
                           color='green', alpha=0.3, label='Gain')
    axes[1, 1].fill_between(days, 0, cumulative_return, 
                           where=(cumulative_return < 0), 
                           color='red', alpha=0.3, label='Loss')
    axes[1, 1].set_title('Cumulative Return from Today', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Days from Now')
    axes[1, 1].set_ylabel('Return (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary(predictions_df, last_price):
    """Print prediction summary statistics"""
    pred_prices = predictions_df['Predicted_Price'].values
    
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    
    print(f"\nCurrent Price:     ${last_price:.2f}")
    print(f"Prediction Period: {len(predictions_df)} days")
    print(f"\nPredicted Prices:")
    print(f"  First Day:       ${pred_prices[0]:.2f}")
    print(f"  Last Day:        ${pred_prices[-1]:.2f}")
    print(f"  Average:         ${pred_prices.mean():.2f}")
    print(f"  Minimum:         ${pred_prices.min():.2f}")
    print(f"  Maximum:         ${pred_prices.max():.2f}")
    print(f"  Std Deviation:   ${pred_prices.std():.2f}")
    
    # Price change analysis
    total_change = pred_prices[-1] - last_price
    total_change_pct = (total_change / last_price) * 100
    
    print(f"\nExpected Change (over {len(predictions_df)} days):")
    print(f"  Absolute:        ${total_change:+.2f}")
    print(f"  Percentage:      {total_change_pct:+.2f}%")
    
    if total_change > 0:
        print(f"  Trend:           📈 BULLISH (Upward)")
    elif total_change < 0:
        print(f"  Trend:           📉 BEARISH (Downward)")
    else:
        print(f"  Trend:           ➡️  NEUTRAL (Sideways)")
    
    # Volatility
    daily_returns = np.diff(pred_prices) / pred_prices[:-1]
    volatility = daily_returns.std() * 100
    print(f"\nPredicted Volatility: {volatility:.2f}%")
    
    print("=" * 70)

def main():
    """Main function to run predictions"""
    print("=" * 70)
    print("LSTM STOCK PRICE PREDICTION - FUTURE FORECAST")
    print("=" * 70)
    
    # Get user input
    try:
        n_days = int(input("\nHow many days to predict? (default: 30): ") or "30")
        if n_days <= 0 or n_days > 365:
            print("Using default: 30 days")
            n_days = 30
    except ValueError:
        print("Invalid input. Using default: 30 days")
        n_days = 30
    
    print("\n")
    
    # Make predictions
    predictions_df, historical_df, last_price = predict_next_days(n_days=n_days)
    
    # Print summary
    print_summary(predictions_df, last_price)
    
    # Show first and last 5 predictions
    print("\nFirst 5 Days:")
    print(predictions_df.head().to_string(index=False))
    
    if len(predictions_df) > 10:
        print("\n...")
        print("\nLast 5 Days:")
        print(predictions_df.tail().to_string(index=False))
    
    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_predictions(predictions_df, historical_df, last_price)
    
    # Save predictions and plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    csv_path = os.path.join(project_dir, "models", "future_predictions.csv")
    plot_path = os.path.join(project_dir, "models", "future_predictions.png")
    
    predictions_df.to_csv(csv_path, index=False)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Predictions saved to: {csv_path}")
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    return predictions_df

if __name__ == "__main__":
    predictions = main()