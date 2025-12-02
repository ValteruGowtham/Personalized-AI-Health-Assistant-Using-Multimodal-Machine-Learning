"""
Medical data preprocessing module with BERT embeddings.
Handles medical history text data and converts to embeddings.
"""
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)


class MedicalDataProcessor:
    """Process medical text data using BERT embeddings."""
    
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16):
        """
        Initialize BERT model and tokenizer.
        
        Args:
            model_name (str): Name of pretrained BERT model
            max_length (int): Maximum sequence length
            batch_size (int): Batch size for processing
        """
        logger.info(f"Loading BERT model: {model_name}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = BertModel.from_pretrained(model_name)
            self.bert_model.eval()  # Set to evaluation mode
            self.max_length = max_length
            self.batch_size = batch_size
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.bert_model.to(self.device)
            logger.info(f"BERT model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise
    
    def preprocess_medical_text(self, texts: List[str]) -> np.ndarray:
        """
        Preprocess medical history text using BERT with batching.
        
        Args:
            texts (list): List of medical history strings
            
        Returns:
            np.array: BERT embeddings (CLS token) of shape (n_samples, 768)
        """
        if not texts:
            raise ValueError("Empty text list provided")
        
        logger.info(f"Processing {len(texts)} medical text records...")
        embeddings = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                
                # Extract CLS token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
                
                if (i + self.batch_size) % 100 == 0:
                    logger.info(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error processing batch {i}-{i+self.batch_size}: {e}")
                raise
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Text processing complete. Shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def load_medical_data(self, file_path: str, text_column: str = 'medical_history') -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load medical history data from CSV and process.
        
        Args:
            file_path (str): Path to CSV file
            text_column (str): Name of column containing medical text
            
        Returns:
            tuple: (embeddings array, original dataframe)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Medical data file not found: {file_path}")
        
        logger.info(f"Loading medical data from {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(data)} records")
            
            # Validate column exists
            if text_column not in data.columns:
                raise ValueError(f"Column '{text_column}' not found. Available columns: {data.columns.tolist()}")
            
            # Check for missing values
            missing_count = data[text_column].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in '{text_column}'. Filling with empty string.")
                data[text_column] = data[text_column].fillna('')
            
            # Extract texts
            texts = data[text_column].tolist()
            
            # Process texts
            embeddings = self.preprocess_medical_text(texts)
            
            return embeddings, data
            
        except Exception as e:
            logger.error(f"Error loading medical data: {e}")
            raise


def load_medical_data(file_path: str, text_column: str = 'medical_history', 
                     batch_size: int = 16) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function to load and process medical data.
    
    Args:
        file_path (str): Path to CSV file
        text_column (str): Name of column containing medical text
        batch_size (int): Batch size for BERT processing
        
    Returns:
        tuple: (embeddings array, original dataframe)
    """
    processor = MedicalDataProcessor(batch_size=batch_size)
    return processor.load_medical_data(file_path, text_column)