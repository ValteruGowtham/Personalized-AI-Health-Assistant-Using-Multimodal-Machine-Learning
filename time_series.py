import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_time_series(file_path):
    """
    Preprocess wearable time-series data.
    Args:
        file_path (str): Path to CSV with time-series data (e.g., heart rate, steps).
    Returns:
        np.array: Scaled time-series data.
    """
    data = pd.read_csv(file_path)
    features = data[['heart_rate', 'steps']].values  # Adjust columns as needed
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    return scaled_data.reshape(-1, features.shape[1], 2)  # Reshape for LSTM/Conv1D

def load_time_series_data(file_path):
    """
    Load and preprocess time-series data.
    Args:
        file_path (str): Path to CSV file.
    Returns:
        np.array: Processed time-series data.
    """
    return preprocess_time_series(file_path)