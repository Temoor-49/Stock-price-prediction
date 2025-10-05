import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocess import load_data as load_data_lr, create_features, scale_and_split
from preprocess_lstm import load_data as load_data_lstm, prepare_lstm_data

def compare_models(sequence_length=60):
    """Compare Linear Regression and LSTM models"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(project_dir, "models")
    
    print("="*70)
    print("MODEL COMPARISON: LINEAR REGRESSION vs LSTM")
    print("="*70)
    
    # ========== Linear Regression ==========
    print("\n[1/2] Evaluating Linear Regression Model...")
    
    # Load and prepare data for Linear Regression
    df_lr = load_data_lr()
    df_feat = create_features(df_lr, n_lags=5)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr, scaler_lr = scale_and_split(df_feat)
    
    # Load Linear Regression model
    lr_model = joblib.load(os.path.join(models_dir, "linear_regression.joblib"))
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test_lr)
    
    # Metrics
    lr_rmse = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
    lr_mae = mean_absolute_error(y_test_lr, y_pred_lr)
    lr_r2 = r2_score(y_test_lr, y_pred_lr)
    
    print(f"  RMSE: {lr_rmse:.4f}")
    print(f"  MAE:  {lr_mae:.4f}")
    print(f"  R²:   {lr_r2:.4f}")
    
    # ========== LSTM ==========
    print("\n[2/2] Evaluating LSTM Model...")
    
    # Load and prepare data for LSTM
    df_lstm = load_data_lstm()
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm = prepare_lstm_data(
        df_lstm, sequence_length=sequence_length
    )
    
    # Load LSTM model
    lstm_model = keras.models.load_model(os.path.join(models_dir, "lstm_model.keras"))
    
    # Predictions
    y_pred_lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    y_pred_lstm = scaler_lstm.inverse_transform(y_pred_lstm_scaled)
    y_test_lstm_actual = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1))
    
    # Metrics
    lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_actual, y_pred_lstm))
    lstm_mae = mean_absolute_error(y_test_lstm_actual, y_pred_lstm)
    lstm_r2 = r2_score(y_test_lstm_actual, y_pred_lstm)
    
    print(f"  RMSE: {lstm_rmse:.4f}")
    print(f"  MAE:  {lstm_mae:.4f}")
    print(f"  R²:   {lstm_r2:.4f}")
    
    # ========== Comparison Summary ==========
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'Linear Regression':<20} {'LSTM':<20} {'Winner':<15}")
    print("-"*70)
    
    # RMSE comparison
    rmse_winner = "LSTM" if lstm_rmse < lr_rmse else "Linear Regression"
    print(f"{'RMSE':<20} {lr_rmse:<20.4f} {lstm_rmse:<20.4f} {rmse_winner:<15}")
    
    # MAE comparison
    mae_winner = "LSTM" if lstm_mae < lr_mae else "Linear Regression"
    print(f"{'MAE':<20} {lr_mae:<20.4f} {lstm_mae:<20.4f} {mae_winner:<15}")
    
    # R² comparison
    r2_winner = "LSTM" if lstm_r2 > lr_r2 else "Linear Regression"
    print(f"{'R²':<20} {lr_r2:<20.4f} {lstm_r2:<20.4f} {r2_winner:<15}")
    
    print("="*70)
    
    # Improvement percentage
    rmse_improvement = ((lr_rmse - lstm_rmse) / lr_rmse) * 100
    mae_improvement = ((lr_mae - lstm_mae) / lr_mae) * 100
    r2_improvement = ((lstm_r2 - lr_r2) / lr_r2) * 100
    
    print("\nLSTM Performance vs Linear Regression:")
    print(f"  RMSE: {rmse_improvement:+.2f}% {'(better)' if rmse_improvement > 0 else '(worse)'}")
    print(f"  MAE:  {mae_improvement:+.2f}% {'(better)' if mae_improvement > 0 else '(worse)'}")
    print(f"  R²:   {r2_improvement:+.2f}% {'(better)' if r2_improvement > 0 else '(worse)'}")
    
    # ========== Visualization ==========
    print("\nCreating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Metrics comparison
    metrics = ['RMSE', 'MAE', 'R²']
    lr_values = [lr_rmse, lr_mae, lr_r2]
    lstm_values = [lstm_rmse, lstm_mae, lstm_r2]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, lr_values, width, label='Linear Regression', alpha=0.8)
    axes[0, 0].bar(x + width/2, lstm_values, width, label='LSTM', alpha=0.8)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Linear Regression predictions
    axes[0, 1].plot(y_test_lr, label='Actual', linewidth=2, alpha=0.7)
    axes[0, 1].plot(y_pred_lr, label='Predicted', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Linear Regression: Test Set Predictions', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: LSTM predictions
    axes[1, 0].plot(y_test_lstm_actual, label='Actual', linewidth=2, alpha=0.7)
    axes[1, 0].plot(y_pred_lstm, label='Predicted', linewidth=2, alpha=0.7)
    axes[1, 0].set_title('LSTM: Test Set Predictions', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error comparison
    lr_errors = y_test_lr - y_pred_lr
    lstm_errors = y_test_lstm_actual.flatten() - y_pred_lstm.flatten()
    
    axes[1, 1].hist(lr_errors, bins=30, alpha=0.6, label='Linear Regression', edgecolor='black')
    axes[1, 1].hist(lstm_errors, bins=30, alpha=0.6, label='LSTM', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Error (Actual - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(models_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.show()
    
    return {
        'linear_regression': {
            'rmse': lr_rmse,
            'mae': lr_mae,
            'r2': lr_r2,
            'predictions': y_pred_lr,
            'actual': y_test_lr
        },
        'lstm': {
            'rmse': lstm_rmse,
            'mae': lstm_mae,
            'r2': lstm_r2,
            'predictions': y_pred_lstm,
            'actual': y_test_lstm_actual
        }
    }

if __name__ == "__main__":
    results = compare_models(sequence_length=60)