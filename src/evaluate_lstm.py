import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocess_lstm import load_data, prepare_lstm_data

def evaluate_lstm_model(sequence_length=60):
    """Evaluate LSTM model and create visualizations"""
    
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load data
    print("Loading data...")
    data_path = os.path.join(project_dir, "data", "aapl_stock.csv")
    df = load_data(data_path)
    
    # Prepare data
    print("Preparing sequences...")
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
        df, sequence_length=sequence_length
    )
    
    # Load model
    print("Loading LSTM model...")
    model_path = os.path.join(project_dir, "models", "lstm_model.keras")
    model = keras.models.load_model(model_path)
    
    # Load scaler
    scaler_path = os.path.join(project_dir, "models", "lstm_scaler.joblib")
    scaler = joblib.load(scaler_path)
    
    # Make predictions
    print("Making predictions...")
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    print("\n" + "="*60)
    print("LSTM MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Root MSE (RMSE):           {test_rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"R² Score:                  {test_r2:.4f}")
    print("="*60 + "\n")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Training predictions
    axes[0, 0].plot(y_train_actual, label='Actual', alpha=0.7)
    axes[0, 0].plot(train_pred, label='Predicted', alpha=0.7)
    axes[0, 0].set_title('Training Set: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test predictions
    axes[0, 1].plot(y_test_actual, label='Actual', linewidth=2)
    axes[0, 1].plot(test_pred, label='Predicted', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Test Set: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction errors
    test_errors = y_test_actual.flatten() - test_pred.flatten()
    axes[1, 0].scatter(range(len(test_errors)), test_errors, alpha=0.6, 
                       c=test_errors, cmap='coolwarm')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Test Set Prediction Errors', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Error (Actual - Predicted)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training history (if available)
    history_path = os.path.join(project_dir, "models", "lstm_history.joblib")
    if os.path.exists(history_path):
        history = joblib.load(history_path)
        axes[1, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[1, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1, 1].set_title('Model Training History', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Training history not available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Training History', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(project_dir, "models", "lstm_evaluation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()
    
    return {
        'rmse': test_rmse,
        'mae': test_mae,
        'r2': test_r2,
        'y_test': y_test_actual,
        'y_pred': test_pred,
        'errors': test_errors
    }

if __name__ == "__main__":
    results = evaluate_lstm_model(sequence_length=60)