"""
Time-series data preprocessing module.
Handles wearable device data (heart rate, steps, sleep, etc.)
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)


class TimeSeriesProcessor:
    """Process time-series data from wearable devices."""
    
    def __init__(self, feature_columns: List[str] = None):
        """
        Initialize time-series processor.
        
        Args:
            feature_columns (list): List of column names to use as features
        """
        self.feature_columns = feature_columns or ['heart_rate', 'steps', 'sleep_hours', 'calories']
        self.scaler = StandardScaler()
        self.is_fitted = False
        logger.info(f"TimeSeriesProcessor initialized with features: {self.feature_columns}")
    
    def preprocess_time_series(self, data: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Preprocess wearable time-series data.
        
        Args:
            data (pd.DataFrame): DataFrame with time-series features
            fit (bool): Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            np.array: Scaled time-series data of shape (n_samples, 1, n_features)
        """
        # Validate columns exist
        missing_cols = set(self.feature_columns) - set(data.columns)
        if missing_cols:
            available_cols = [col for col in self.feature_columns if col in data.columns]
            if not available_cols:
                raise ValueError(f"None of the required columns found. Required: {self.feature_columns}, Available: {data.columns.tolist()}")
            logger.warning(f"Missing columns {missing_cols}. Using available: {available_cols}")
            self.feature_columns = available_cols
        
        # Extract features
        features = data[self.feature_columns].values
        logger.info(f"Extracted features shape: {features.shape}")
        
        # Check for missing values
        if np.isnan(features).any():
            logger.warning("Found NaN values in time-series data. Filling with column means.")
            features = pd.DataFrame(features).fillna(pd.DataFrame(features).mean()).values
        
        # Scale features
        if fit:
            scaled_data = self.scaler.fit_transform(features)
            self.is_fitted = True
            logger.info("Scaler fitted to data")
        else:
            if not self.is_fitted:
                logger.warning("Scaler not fitted yet. Fitting now.")
                scaled_data = self.scaler.fit_transform(features)
                self.is_fitted = True
            else:
                scaled_data = self.scaler.transform(features)
        
        # Reshape for Conv1D/LSTM: (n_samples, timesteps, features)
        # We treat each sample as a single timestep with multiple features
        reshaped_data = scaled_data.reshape(-1, 1, len(self.feature_columns))
        logger.info(f"Reshaped data shape: {reshaped_data.shape}")
        
        return reshaped_data
    
    def load_time_series_data(self, file_path: str, fit: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load and preprocess time-series data from CSV.
        
        Args:
            file_path (str): Path to CSV file
            fit (bool): Whether to fit the scaler
            
        Returns:
            tuple: (processed array, original dataframe)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Time-series data file not found: {file_path}")
        
        logger.info(f"Loading time-series data from {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} records with columns: {data.columns.tolist()}")
            
            # Process data
            processed_data = self.preprocess_time_series(data, fit=fit)
            
            return processed_data, data
            
        except Exception as e:
            logger.error(f"Error loading time-series data: {e}")
            raise


def load_time_series_data(file_path: str, 
                         feature_columns: List[str] = None,
                         fit: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function to load and process time-series data.
    
    Args:
        file_path (str): Path to CSV file
        feature_columns (list): List of column names to use as features
        fit (bool): Whether to fit the scaler
        
    Returns:
        tuple: (processed array, original dataframe)
    """
    processor = TimeSeriesProcessor(feature_columns=feature_columns)
    return processor.load_time_series_data(file_path, fit=fit)