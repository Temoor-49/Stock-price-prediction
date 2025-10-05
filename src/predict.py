import joblib
import pandas as pd
from preprocess import load_data, create_features, scale_and_split

def predict_next(n_lags=5):
    # Load full dataset
    df = load_data()
    df_feat = create_features(df, n_lags=n_lags)

    # Take last available row as input
    last_row = df_feat.drop("Close", axis=1).iloc[-1].values.reshape(1, -1)

    # Load scaler & transform
    scaler = joblib.load("models/scaler.joblib")
    last_scaled = scaler.transform(last_row)

    # Load model
    model = joblib.load("models/linear_regression.joblib")

    # Predict
    prediction = model.predict(last_scaled)[0]
    print(f"Predicted next day close price: ${prediction:.2f}")
    return prediction

if __name__ == "__main__":
    predict_next()
