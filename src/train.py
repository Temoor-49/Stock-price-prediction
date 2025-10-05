import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_data, create_features, scale_and_split

def train_model():
    # Get script directory for building paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load and preprocess
    data_path = os.path.join(project_dir, "data", "aapl_stock.csv")
    df = load_data(data_path)
    df_feat = create_features(df, n_lags=5)
    X_train, X_test, y_train, y_test, scaler = scale_and_split(df_feat)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R²:", r2)

    # Save model & scaler (with proper paths)
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)  # Create models dir if it doesn't exist
    
    joblib.dump(model, os.path.join(models_dir, "linear_regression.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    print(f"Model and scaler saved to {models_dir}")

if __name__ == "__main__":
    train_model()