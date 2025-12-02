"""
Utility functions for the health assistant project.
"""
import os
import yaml
import logging
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def setup_logging(config):
    """
    Setup logging configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    log_level = getattr(logging, config['logging']['level'])
    log_file = config['logging']['log_file']
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_directories(config):
    """
    Create necessary directories for outputs and models.
    
    Args:
        config (dict): Configuration dictionary
    """
    dirs = [
        config['data']['output_dir'],
        config['data']['models_dir'],
        config['explainability']['plot_dir']
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_model_metadata(model, config, metrics, filepath):
    """
    Save model metadata and training information.
    
    Args:
        model: Trained model
        config (dict): Configuration used
        metrics (dict): Training metrics
        filepath (str): Path to save metadata
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics,
        'model_summary': []
    }
    
    # Save metadata
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def plot_training_history(history, save_path=None):
    """
    Plot training history with loss and accuracy.
    
    Args:
        history: Keras training history object
        save_path (str): Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='black')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path (str): Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    return roc_auc


def evaluate_model(model, X_test, y_test, save_dir=None):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features (list of arrays for multimodal)
        y_test: Test labels
        save_dir (str): Directory to save evaluation plots
        
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, 
                            os.path.join(save_dir, 'confusion_matrix.png'))
        
        # Plot ROC curve
        roc_auc = plot_roc_curve(y_test, y_pred_proba, 
                                os.path.join(save_dir, 'roc_curve.png'))
        metrics['roc_auc'] = float(roc_auc)
        
        # Save metrics
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


def validate_data(df, required_columns):
    """
    Validate that dataframe has required columns.
    
    Args:
        df: Pandas dataframe
        required_columns (list): List of required column names
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def print_metrics_summary(metrics):
    """
    Print formatted metrics summary.
    
    Args:
        metrics (dict): Dictionary of metrics
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key.upper():20s}: {value:.4f}")
    print("="*60 + "\n")
