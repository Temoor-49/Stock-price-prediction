import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pandas as pd
import os
from preprocess import load_data, create_features, scale_and_split

def evaluate_model():
    """Evaluate the trained model and visualize results"""
    
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Reload data
    print("Loading data...")
    data_path = os.path.join(project_dir, "data", "aapl_stock.csv")
    df = load_data(data_path)
    
    # Debug: Check data types
    print(f"Data shape: {df.shape}")
    print(f"Close column dtype: {df['Close'].dtype}")
    
    df_feat = create_features(df, n_lags=5)
    X_train, X_test, y_train, y_test, scaler = scale_and_split(df_feat)
    
    # Ensure arrays are numeric
    y_test = y_test.astype(float)
    y_train = y_train.astype(float)
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(project_dir, "models", "linear_regression.joblib")
    model = joblib.load(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Squared Error (MSE):  {mse:.4f}")
    print(f"Root MSE (RMSE):           {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score:                  {r2:.4f}")
    print("="*50 + "\n")
    
    # Create visualization
    plt.figure(figsize=(14,6))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.plot(y_test, label="Actual", linewidth=2, marker='o', markersize=3)
    plt.plot(y_pred, label="Predicted", linewidth=2, marker='s', markersize=3, alpha=0.7)
    plt.title("Stock Price: Actual vs Predicted", fontsize=14, fontweight='bold')
    plt.xlabel("Test Sample Index", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Error
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred
    plt.scatter(range(len(errors)), errors, alpha=0.6, c=errors, cmap='coolwarm')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.title("Prediction Errors", fontsize=14, fontweight='bold')
    plt.xlabel("Test Sample Index", fontsize=12)
    plt.ylabel("Error (Actual - Predicted)", fontsize=12)
    plt.colorbar(label='Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(project_dir, "models", "evaluation_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }

if __name__ == "__main__":
    results = evaluate_model()