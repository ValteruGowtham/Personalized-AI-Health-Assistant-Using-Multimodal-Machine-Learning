import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def preprocess_medical_text(texts):
    """
    Preprocess medical history text using BERT.
    Args:
        texts (list): List of medical history strings.
    Returns:
        np.array: BERT embeddings (CLS token).
    """
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())  # CLS token
    return np.array(embeddings).squeeze()

def load_medical_data(file_path):
    """
    Load medical history data from CSV.
    Args:
        file_path (str): Path to CSV file.
    Returns:
        np.array: Processed BERT embeddings.
    """
    data = pd.read_csv(file_path)
    texts = data['medical_history'].tolist()  # Adjust column name as needed
    return preprocess_medical_text(texts)