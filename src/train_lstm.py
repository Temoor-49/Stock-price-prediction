import numpy as np
import joblib
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from preprocess_lstm import load_data, prepare_lstm_data

def build_lstm_model(input_shape, units=[50, 50], dropout=0.2):
    """Build LSTM model."""
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=units[0], return_sequences=len(units) > 1, 
                   input_shape=input_shape))
    model.add(Dropout(dropout))
    
    # Additional LSTM layers
    for i in range(1, len(units)):
        return_seq = i < len(units) - 1
        model.add(LSTM(units=units[i], return_sequences=return_seq))
        model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(sequence_length=60, epochs=50, batch_size=32):
    """Train LSTM model for stock price prediction"""
    
    # Get project directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load and preprocess data
    print("Loading data...")
    data_path = os.path.join(project_dir, "data", "aapl_stock.csv")
    df = load_data(data_path)
    print(f"Data shape: {df.shape}")
    
    # Prepare LSTM data
    print(f"\nPreparing sequences (sequence_length={sequence_length})...")
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
        df, sequence_length=sequence_length
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=[50, 50],
        dropout=0.2
    )
    
    print(model.summary())
    
    # Callbacks
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(models_dir, "lstm_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    print(f"\nTraining model for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Inverse transform to get actual prices
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
    train_mae = mean_absolute_error(y_train_actual, train_pred)
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    print("\nTraining Set Metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*60)
    
    # Save model and scaler
    model_path = os.path.join(models_dir, "lstm_model.keras")
    scaler_path = os.path.join(models_dir, "lstm_scaler.joblib")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save training history
    history_path = os.path.join(models_dir, "lstm_history.joblib")
    joblib.dump(history.history, history_path)
    print(f"✓ Training history saved to: {history_path}")
    
    return model, history, scaler

if __name__ == "__main__":
    model, history, scaler = train_lstm_model(
        sequence_length=60,
        epochs=50,
        batch_size=32
    )