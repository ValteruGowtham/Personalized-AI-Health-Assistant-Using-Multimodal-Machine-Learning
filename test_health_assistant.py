"""
Unit tests for the health assistant project.
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch

# Import modules to test
from utils import load_config, validate_data
from time_series import TimeSeriesProcessor
from data_generator import generate_medical_history_data, generate_wearable_data


class TestUtils:
    """Test utility functions."""
    
    def test_load_config(self):
        """Test configuration loading."""
        config = load_config('config.yaml')
        assert config is not None
        assert 'data' in config
        assert 'model' in config
        assert 'training' in config
    
    def test_validate_data(self):
        """Test data validation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Should not raise error
        validate_data(df, ['col1', 'col2'])
        
        # Should raise error for missing columns
        with pytest.raises(ValueError):
            validate_data(df, ['col1', 'col3'])


class TestDataGenerator:
    """Test data generation functions."""
    
    def test_generate_medical_history_data(self):
        """Test medical history data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_medical.csv')
            df = generate_medical_history_data(num_samples=100, output_path=output_path)
            
            assert len(df) == 100
            assert 'patient_id' in df.columns
            assert 'medical_history' in df.columns
            assert 'disease_label' in df.columns
            assert os.path.exists(output_path)
    
    def test_generate_wearable_data(self):
        """Test wearable data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_wearable.csv')
            df = generate_wearable_data(num_samples=100, output_path=output_path)
            
            assert len(df) == 100
            assert 'patient_id' in df.columns
            assert 'heart_rate' in df.columns
            assert 'steps' in df.columns
            assert os.path.exists(output_path)


class TestTimeSeriesProcessor:
    """Test time-series processing."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = TimeSeriesProcessor(feature_columns=['heart_rate', 'steps'])
        assert processor.feature_columns == ['heart_rate', 'steps']
        assert not processor.is_fitted
    
    def test_preprocess_time_series(self):
        """Test time-series preprocessing."""
        processor = TimeSeriesProcessor(feature_columns=['heart_rate', 'steps'])
        
        # Create sample data
        data = pd.DataFrame({
            'heart_rate': [70, 75, 80, 72, 68],
            'steps': [8000, 9000, 7500, 8500, 7000]
        })
        
        # Process data
        processed = processor.preprocess_time_series(data, fit=True)
        
        assert processed.shape == (5, 1, 2)  # (samples, timesteps, features)
        assert processor.is_fitted
    
    def test_missing_columns_handling(self):
        """Test handling of missing columns."""
        processor = TimeSeriesProcessor(feature_columns=['heart_rate', 'steps', 'missing_col'])
        
        data = pd.DataFrame({
            'heart_rate': [70, 75],
            'steps': [8000, 9000]
        })
        
        # Should handle missing columns gracefully
        processed = processor.preprocess_time_series(data, fit=True)
        assert processed.shape[2] == 2  # Only 2 features available


class TestMedicalDataProcessor:
    """Test medical data processing."""
    
    @patch('medical_data.BertTokenizer')
    @patch('medical_data.BertModel')
    def test_initialization(self, mock_bert_model, mock_tokenizer):
        """Test processor initialization."""
        from medical_data import MedicalDataProcessor
        
        processor = MedicalDataProcessor(batch_size=8)
        assert processor.batch_size == 8
        assert processor.max_length == 128


class TestFusionModel:
    """Test fusion model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        from fusion_model import MultimodalFusionModel
        
        config = load_config('config.yaml')
        model = MultimodalFusionModel(config)
        
        assert model.config == config
        assert model.model is None
        assert model.history is None
    
    def test_model_building(self):
        """Test model building."""
        from fusion_model import MultimodalFusionModel
        
        config = load_config('config.yaml')
        fusion_model = MultimodalFusionModel(config)
        
        text_shape = (768,)
        ts_shape = (1, 4)
        
        model = fusion_model.build_model(text_shape, ts_shape)
        
        assert model is not None
        assert len(model.inputs) == 2
        assert len(model.outputs) == 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
