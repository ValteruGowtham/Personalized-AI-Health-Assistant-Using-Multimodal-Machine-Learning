"""
Multimodal fusion model for health prediction.
Combines text (BERT), time-series, and image data.
"""
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import (Input, Dense, Concatenate, Conv1D, LSTM, 
                                     Dropout, BatchNormalization, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import shap
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class MultimodalFusionModel:
    """Multimodal fusion model for health prediction."""
    
    def __init__(self, config: dict):
        """
        Initialize fusion model with configuration.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        logger.info("MultimodalFusionModel initialized")
    
    def build_model(self, text_shape: Tuple[int], ts_shape: Tuple[int], 
                   image_shape: Optional[Tuple[int]] = None) -> Model:
        """
        Build multimodal fusion model.
        
        Args:
            text_shape (tuple): Shape of BERT embeddings (embedding_dim,)
            ts_shape (tuple): Shape of time-series data (timesteps, features)
            image_shape (tuple): Optional shape of image data
            
        Returns:
            Model: Compiled Keras model
        """
        logger.info("Building multimodal fusion model...")
        logger.info(f"Text shape: {text_shape}, TS shape: {ts_shape}, Image shape: {image_shape}")
        
        model_config = self.config['model']
        
        # Text branch (BERT embeddings)
        text_input = Input(shape=text_shape, name='text_input')
        text_dense = Dense(model_config['text_dense_units'], activation='relu')(text_input)
        text_dense = BatchNormalization()(text_dense)
        text_dense = Dropout(model_config['dropout_rate'])(text_dense)
        text_dense = Dense(64, activation='relu')(text_dense)
        
        # Time-series branch
        ts_input = Input(shape=ts_shape, name='ts_input')
        ts_conv = Conv1D(model_config['ts_conv_filters'], 
                        kernel_size=model_config['ts_conv_kernel'], 
                        activation='relu', padding='same')(ts_input)
        ts_conv = BatchNormalization()(ts_conv)
        ts_lstm = LSTM(model_config['ts_lstm_units'], return_sequences=True)(ts_conv)
        ts_lstm = Dropout(model_config['dropout_rate'])(ts_lstm)
        ts_pool = GlobalAveragePooling1D()(ts_lstm)
        ts_dense = Dense(64, activation='relu')(ts_pool)
        
        # Combine branches
        inputs = [text_input, ts_input]
        branches = [text_dense, ts_dense]
        
        # Image branch (optional)
        if image_shape is not None:
            image_input = Input(shape=image_shape, name='image_input')
            image_dense = Dense(128, activation='relu')(image_input)
            image_dense = BatchNormalization()(image_dense)
            image_dense = Dropout(model_config['dropout_rate'])(image_dense)
            image_dense = Dense(64, activation='relu')(image_dense)
            inputs.append(image_input)
            branches.append(image_dense)
            logger.info("Image branch added to model")
        
        # Fusion layer
        if len(branches) > 1:
            concat = Concatenate()(branches)
        else:
            concat = branches[0]
        
        # Dense layers
        dense = Dense(model_config['fusion_dense_units'], activation='relu')(concat)
        dense = BatchNormalization()(dense)
        dense = Dropout(model_config['dropout_rate'])(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dropout(model_config['dropout_rate'] / 2)(dense)
        
        # Output layer (binary classification)
        output = Dense(1, activation='sigmoid', name='output')(dense)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=output, name='multimodal_health_model')
        
        optimizer = Adam(learning_rate=model_config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        self.model = model
        logger.info("Model built successfully")
        logger.info(f"Model has {model.count_params():,} trainable parameters")
        
        return model
    
    def get_callbacks(self, model_path: str) -> List:
        """
        Get training callbacks.
        
        Args:
            model_path (str): Path to save best model
            
        Returns:
            list: List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train: List[np.ndarray], y_train: np.ndarray,
             X_val: List[np.ndarray], y_val: np.ndarray,
             model_path: str) -> dict:
        """
        Train the fusion model.
        
        Args:
            X_train (list): List of training data arrays [text, ts, image]
            y_train (np.array): Training labels
            X_val (list): List of validation data arrays
            y_val (np.array): Validation labels
            model_path (str): Path to save model
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info("Starting model training...")
        logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
        
        # Get callbacks
        callbacks = self.get_callbacks(model_path)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training complete!")
        
        return self.history.history
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath (str): Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath (str): Path to model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = keras_load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (list): List of input arrays
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded or built")
        
        return self.model.predict(X)
    
    def explain_predictions(self, X: List[np.ndarray], 
                          num_samples: int = 100,
                          save_dir: Optional[str] = None):
        """
        Explain predictions using SHAP.
        
        Args:
            X (list): List of input arrays
            num_samples (int): Number of samples for SHAP
            save_dir (str): Directory to save plots
        """
        if self.model is None:
            raise ValueError("Model not loaded or built")
        
        logger.info(f"Generating SHAP explanations for {num_samples} samples...")
        
        # Limit samples
        X_sample = [x[:num_samples] for x in X]
        
        try:
            # Create SHAP explainer
            explainer = shap.KernelExplainer(
                lambda x: self.model.predict(self._reshape_shap_input(x, X_sample)),
                self._flatten_inputs(X_sample[:10])  # Use 10 background samples
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(self._flatten_inputs(X_sample))
            
            # Create feature names
            feature_names = self._get_feature_names(X)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, self._flatten_inputs(X_sample), 
                            feature_names=feature_names, show=False)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, 'shap_summary.png'), 
                          dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_dir}")
            
            plt.close()
            
            logger.info("SHAP explanation complete")
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            logger.warning("SHAP explanation failed, but model is still functional")
    
    def _flatten_inputs(self, X: List[np.ndarray]) -> np.ndarray:
        """Flatten multimodal inputs for SHAP."""
        flattened = []
        for x in X:
            if len(x.shape) > 2:
                flattened.append(x.reshape(x.shape[0], -1))
            else:
                flattened.append(x)
        return np.concatenate(flattened, axis=1)
    
    def _reshape_shap_input(self, flat_input: np.ndarray, 
                           original_X: List[np.ndarray]) -> List[np.ndarray]:
        """Reshape flattened SHAP input back to original shapes."""
        reshaped = []
        idx = 0
        for x in original_X:
            size = np.prod(x.shape[1:])
            reshaped.append(flat_input[:, idx:idx+size].reshape(-1, *x.shape[1:]))
            idx += size
        return reshaped
    
    def _get_feature_names(self, X: List[np.ndarray]) -> List[str]:
        """Generate feature names for visualization."""
        names = []
        names.extend([f'text_{i}' for i in range(X[0].shape[1])])
        if len(X) > 1:
            names.extend([f'ts_{i}' for i in range(np.prod(X[1].shape[1:]))])
        if len(X) > 2:
            names.extend([f'image_{i}' for i in range(np.prod(X[2].shape[1:]))])
        return names


# Convenience functions
def build_fusion_model(config: dict, text_shape: Tuple[int], 
                      ts_shape: Tuple[int],
                      image_shape: Optional[Tuple[int]] = None) -> MultimodalFusionModel:
    """
    Build and return fusion model.
    
    Args:
        config (dict): Configuration dictionary
        text_shape (tuple): Shape of text embeddings
        ts_shape (tuple): Shape of time-series data
        image_shape (tuple): Optional shape of image data
        
    Returns:
        MultimodalFusionModel: Built model
    """
    fusion_model = MultimodalFusionModel(config)
    fusion_model.build_model(text_shape, ts_shape, image_shape)
    return fusion_model