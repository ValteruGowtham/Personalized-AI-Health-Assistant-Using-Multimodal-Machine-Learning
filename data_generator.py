"""
Generate synthetic medical data for testing and demonstration.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_medical_history_data(num_samples=1000, output_path='data/medical_history.csv'):
    """
    Generate synthetic medical history data.
    
    Args:
        num_samples (int): Number of samples to generate
        output_path (str): Path to save CSV file
    """
    np.random.seed(42)
    
    # Medical conditions and symptoms
    conditions = [
        "hypertension", "diabetes", "asthma", "arthritis", "none"
    ]
    
    symptoms = [
        "chest pain", "shortness of breath", "fatigue", "dizziness",
        "headache", "nausea", "joint pain", "no symptoms"
    ]
    
    medications = [
        "aspirin", "metformin", "lisinopril", "albuterol", "none"
    ]
    
    data = []
    for i in range(num_samples):
        # Generate patient profile
        age = np.random.randint(25, 85)
        gender = np.random.choice(['Male', 'Female'])
        
        # Generate medical history text
        num_conditions = np.random.randint(0, 3)
        patient_conditions = np.random.choice(conditions, size=num_conditions, replace=False)
        
        num_symptoms = np.random.randint(0, 4)
        patient_symptoms = np.random.choice(symptoms, size=num_symptoms, replace=False)
        
        num_meds = np.random.randint(0, 3)
        patient_meds = np.random.choice(medications, size=num_meds, replace=False)
        
        # Create medical history text
        history_parts = []
        history_parts.append(f"{age} year old {gender.lower()} patient")
        
        if len(patient_conditions) > 0:
            history_parts.append(f"with history of {', '.join(patient_conditions)}")
        
        if len(patient_symptoms) > 0:
            history_parts.append(f"presenting with {', '.join(patient_symptoms)}")
        
        if len(patient_meds) > 0:
            history_parts.append(f"currently taking {', '.join(patient_meds)}")
        
        medical_history = ". ".join(history_parts) + "."
        
        # Generate disease label (binary: 0 = healthy, 1 = disease)
        # Higher probability of disease with age and conditions
        disease_prob = 0.2 + (age / 200) + (len(patient_conditions) * 0.15)
        disease_label = 1 if np.random.random() < disease_prob else 0
        
        data.append({
            'patient_id': f'P{i:05d}',
            'age': age,
            'gender': gender,
            'medical_history': medical_history,
            'disease_label': disease_label
        })
    
    # Create dataframe and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} medical history records at {output_path}")
    print(f"Disease distribution: {df['disease_label'].value_counts().to_dict()}")
    
    return df


def generate_wearable_data(num_samples=1000, output_path='data/wearable_data.csv'):
    """
    Generate synthetic wearable device time-series data.
    
    Args:
        num_samples (int): Number of samples to generate
        output_path (str): Path to save CSV file
    """
    np.random.seed(42)
    
    data = []
    for i in range(num_samples):
        # Generate time-series features (7 days of data)
        days = 7
        
        # Healthy vs unhealthy patterns
        is_healthy = np.random.random() > 0.4
        
        if is_healthy:
            # Healthy patterns
            heart_rate = np.random.normal(70, 8, days)
            steps = np.random.normal(8000, 2000, days)
            sleep_hours = np.random.normal(7.5, 1, days)
            calories = np.random.normal(2000, 300, days)
        else:
            # Unhealthy patterns
            heart_rate = np.random.normal(85, 12, days)
            steps = np.random.normal(4000, 1500, days)
            sleep_hours = np.random.normal(5.5, 1.5, days)
            calories = np.random.normal(2500, 400, days)
        
        # Ensure realistic bounds
        heart_rate = np.clip(heart_rate, 50, 120)
        steps = np.clip(steps, 0, 20000)
        sleep_hours = np.clip(sleep_hours, 3, 12)
        calories = np.clip(calories, 1200, 4000)
        
        # Average values for the week
        data.append({
            'patient_id': f'P{i:05d}',
            'heart_rate': heart_rate.mean(),
            'steps': steps.mean(),
            'sleep_hours': sleep_hours.mean(),
            'calories': calories.mean(),
            'heart_rate_std': heart_rate.std(),
            'steps_std': steps.std()
        })
    
    # Create dataframe and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} wearable data records at {output_path}")
    
    return df


def merge_datasets(medical_path='data/medical_history.csv', 
                   wearable_path='data/wearable_data.csv',
                   output_path='data/combined_data.csv'):
    """
    Merge medical history and wearable data on patient_id.
    
    Args:
        medical_path (str): Path to medical history CSV
        wearable_path (str): Path to wearable data CSV
        output_path (str): Path to save combined CSV
    """
    medical_df = pd.read_csv(medical_path)
    wearable_df = pd.read_csv(wearable_path)
    
    # Merge on patient_id
    combined_df = pd.merge(medical_df, wearable_df, on='patient_id', how='inner')
    
    combined_df.to_csv(output_path, index=False)
    print(f"Merged datasets saved to {output_path}")
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df


if __name__ == "__main__":
    print("Generating synthetic medical data...")
    print("="*60)
    
    # Generate datasets
    medical_df = generate_medical_history_data(num_samples=1000)
    wearable_df = generate_wearable_data(num_samples=1000)
    combined_df = merge_datasets()
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)
    print("\nSample medical history:")
    print(medical_df.head(3))
    print("\nSample wearable data:")
    print(wearable_df.head(3))
