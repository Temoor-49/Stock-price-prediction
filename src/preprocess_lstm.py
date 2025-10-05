import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(path=None):
    """Load stock CSV data"""
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
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

def create_sequences(data, sequence_length=60):
    """
    Create sequences for LSTM training.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        sequence_length: number of time steps to look back
    
    Returns:
        X: sequences of shape (n_samples, sequence_length, n_features)
        y: targets of shape (n_samples,)
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # Predict the 'Close' price (first column)
    
    return np.array(X), np.array(y)

def prepare_lstm_data(df, sequence_length=60, train_split=0.8, features=['Close']):
    """
    Prepare data for LSTM model.
    
    Args:
        df: pandas DataFrame with stock data
        sequence_length: number of time steps for LSTM
        train_split: fraction of data to use for training
        features: list of column names to use as features
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Select features
    data = df[features].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split into train and test
    train_size = int(len(scaled_data) * train_split)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]  # Include lookback for test
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    
    return X_train, X_test, y_train, y_test, scaler

def prepare_multivariate_lstm_data(df, sequence_length=60, train_split=0.8):
    """
    Prepare multivariate data for LSTM (using multiple features).
    
    Args:
        df: pandas DataFrame with stock data
        sequence_length: number of time steps for LSTM
        train_split: fraction of data to use for training
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Use multiple features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_features = [f for f in features if f in df.columns]
    
    return prepare_lstm_data(df, sequence_length, train_split, available_features)