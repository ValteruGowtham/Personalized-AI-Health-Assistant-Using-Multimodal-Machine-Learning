"""
Main training pipeline for Personalized AI Health Assistant.
Orchestrates data loading, model training, evaluation, and visualization.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Import project modules
from utils import (load_config, setup_logging, create_directories, 
                  plot_training_history, evaluate_model, print_metrics_summary,
                  save_model_metadata)
from medical_data import MedicalDataProcessor
from time_series import TimeSeriesProcessor
from fusion_model import MultimodalFusionModel


def main():
    """Main training pipeline."""
    
    # Load configuration
    print("="*80)
    print("PERSONALIZED AI HEALTH ASSISTANT - TRAINING PIPELINE")
    print("="*80)
    
    config = load_config('config.yaml')
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting health assistant training pipeline")
    
    # Create necessary directories
    create_directories(config)
    
    try:
        # ============================================================
        # STEP 1: Load and Process Data
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 1: LOADING AND PROCESSING DATA")
        logger.info("="*60)
        
        # Load medical text data
        logger.info("\nProcessing medical history data...")
        medical_processor = MedicalDataProcessor(batch_size=16)
        text_embeddings, medical_df = medical_processor.load_medical_data(
            config['data']['medical_history'],
            text_column=config['columns']['medical_history']
        )
        
        # Load time-series data
        logger.info("\nProcessing wearable time-series data...")
        ts_processor = TimeSeriesProcessor(
            feature_columns=config['columns']['time_series']
        )
        ts_data, wearable_df = ts_processor.load_time_series_data(
            config['data']['wearable_data'],
            fit=True
        )
        
        # Get labels
        if config['columns']['target'] not in medical_df.columns:
            raise ValueError(f"Target column '{config['columns']['target']}' not found in data")
        
        labels = medical_df[config['columns']['target']].values
        logger.info(f"\nLabels shape: {labels.shape}")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        # ============================================================
        # STEP 2: Split Data
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 2: SPLITTING DATA")
        logger.info("="*60)
        
        # First split: separate test set
        test_size = config['training']['test_split']
        val_size = config['training']['validation_split']
        
        # Split into train+val and test
        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=config['training']['random_seed'],
            stratify=labels
        )
        
        # Split train+val into train and val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=config['training']['random_seed'],
            stratify=labels[train_val_idx]
        )
        
        # Create data splits
        X_train = [text_embeddings[train_idx], ts_data[train_idx]]
        y_train = labels[train_idx]
        
        X_val = [text_embeddings[val_idx], ts_data[val_idx]]
        y_val = labels[val_idx]
        
        X_test = [text_embeddings[test_idx], ts_data[test_idx]]
        y_test = labels[test_idx]
        
        logger.info(f"Training samples: {len(y_train)}")
        logger.info(f"Validation samples: {len(y_val)}")
        logger.info(f"Test samples: {len(y_test)}")
        logger.info(f"Train class distribution: {np.bincount(y_train)}")
        logger.info(f"Val class distribution: {np.bincount(y_val)}")
        logger.info(f"Test class distribution: {np.bincount(y_test)}")
        
        # ============================================================
        # STEP 3: Build Model
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 3: BUILDING MODEL")
        logger.info("="*60)
        
        fusion_model = MultimodalFusionModel(config)
        
        # Get shapes
        text_shape = (text_embeddings.shape[1],)
        ts_shape = ts_data.shape[1:]
        
        logger.info(f"Text embedding shape: {text_shape}")
        logger.info(f"Time-series shape: {ts_shape}")
        
        # Build model
        model = fusion_model.build_model(text_shape, ts_shape, image_shape=None)
        
        # Print model summary
        logger.info("\nModel Architecture:")
        model.summary(print_fn=logger.info)
        
        # ============================================================
        # STEP 4: Train Model
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 4: TRAINING MODEL")
        logger.info("="*60)
        
        model_path = os.path.join(config['data']['models_dir'], 'best_model.keras')
        
        history = fusion_model.train(
            X_train, y_train,
            X_val, y_val,
            model_path=model_path
        )
        
        # Plot training history
        history_plot_path = os.path.join(config['data']['output_dir'], 'training_history.png')
        plot_training_history(fusion_model.history, save_path=history_plot_path)
        
        # ============================================================
        # STEP 5: Evaluate Model
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 5: EVALUATING MODEL")
        logger.info("="*60)
        
        eval_dir = os.path.join(config['data']['output_dir'], 'evaluation')
        metrics = evaluate_model(model, X_test, y_test, save_dir=eval_dir)
        
        print_metrics_summary(metrics)
        
        # ============================================================
        # STEP 6: Save Model and Metadata
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 6: SAVING MODEL AND METADATA")
        logger.info("="*60)
        
        final_model_path = os.path.join(config['data']['models_dir'], 'final_model.keras')
        fusion_model.save_model(final_model_path)
        
        # Save metadata
        metadata_path = os.path.join(config['data']['models_dir'], 'model_metadata.json')
        save_model_metadata(model, config, metrics, metadata_path)
        
        logger.info(f"Model saved to: {final_model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # ============================================================
        # STEP 7: Generate Explanations
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 7: GENERATING MODEL EXPLANATIONS")
        logger.info("="*60)
        
        try:
            fusion_model.explain_predictions(
                X_test,
                num_samples=config['explainability']['num_samples'],
                save_dir=config['explainability']['plot_dir']
            )
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            logger.info("Continuing without SHAP explanations...")
        
        # ============================================================
        # COMPLETION
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {config['data']['output_dir']}")
        logger.info(f"Model saved to: {config['data']['models_dir']}")
        logger.info(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Final Test F1-Score: {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"Final Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE! Check the 'outputs' directory for results.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        print(f"\n❌ Training failed: {str(e)}")
        print("Check the log file for details.")
        raise


if __name__ == "__main__":
    main()