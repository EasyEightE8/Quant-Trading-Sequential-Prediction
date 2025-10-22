import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Dict
import os
from pathlib import Path

# Configuration constants
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / 'api' / 'models' / 'best_lstm_model.h5'
SCALER_PATH = BASE_DIR / 'api' / 'models' / 'scaler.pkl'
WINDOW_SIZE = 30
PREDICTION_THRESHOLD = 0.40

# Global variables for model and scaler
model = None
scaler = None

# Load model and scaler at module load time
def load_attributes():
    global model, scaler
    
    # Load the pre-trained model
    try:
        tf.get_logger().setLevel('ERROR') 
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    
    # Load the scaler
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        scaler = None

# Initialize model and scaler
def predict_signals(input_data: List[Dict]):
    if model is None or scaler is None:
        return 0.5, 0
    
    # Convert input data to DataFrame
    df_input = pd.DataFrame(input_data)
    
    # Validate input length
    if len(df_input) != WINDOW_SIZE:
        raise ValueError(f"Input data must contain exactly {WINDOW_SIZE} records.")
    
    # Scale the data
    data_scaled = scaler.transform(df_input)
    
    # Reshape data for LSTM input
    data_3d = data_scaled[np.newaxis, :, :]
    
    # Make prediction
    prediction_proba = model.predict(data_3d, verbose=0)[0][0]
    
    # Determine signal based on threshold
    signal = 1 if prediction_proba > PREDICTION_THRESHOLD else 0

    return float(prediction_proba), int(signal)