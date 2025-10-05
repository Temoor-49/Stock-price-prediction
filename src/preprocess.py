import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_data(path=None):
    """Load stock CSV data"""
    if path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build path relative to script location
        path = os.path.join(script_dir, "..", "data", "aapl_stock.csv")
    
    df = pd.read_csv(path)
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values after conversion
    df = df.dropna()
    
    return df

def create_features(df, n_lags=5):
    """
    Create lag features for supervised learning.
    Example: use last 5 days' closing prices to predict next day.
    """
    df_feat = pd.DataFrame(index=df.index)
    df_feat["Close"] = df["Close"]

    # Create lag features
    for lag in range(1, n_lags+1):
        df_feat[f"lag_{lag}"] = df["Close"].shift(lag)

    # Drop rows with NaN (from shifting)
    df_feat = df_feat.dropna()
    return df_feat

def scale_and_split(df_feat, test_size=0.2):
    """Scale features and split into train/test"""
    X = df_feat.drop("Close", axis=1).values
    y = df_feat["Close"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test, scaler