"""
Streamlit Web UI for Personalized AI Health Assistant.
Provides an interactive interface for health predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# Import project modules
from utils import load_config
from medical_data import MedicalDataProcessor
from time_series import TimeSeriesProcessor
from fusion_model import MultimodalFusionModel


# Page configuration
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load configuration and trained model."""
    try:
        config = load_config('config.yaml')
        fusion_model = MultimodalFusionModel(config)
        
        model_path = os.path.join(config['data']['models_dir'], 'final_model.keras')
        if os.path.exists(model_path):
            fusion_model.load_model(model_path)
            return fusion_model, config, None
        else:
            return None, config, "Model not found. Please train the model first."
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Health Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Personalized Health Predictions Using Multimodal Machine Learning")
    st.markdown("---")
    
    # Load model
    fusion_model, config, error = load_model_and_config()
    
    if error:
        st.error(error)
        st.info("💡 Run `python main.py` to train the model first, or `python data_generator.py` to generate sample data.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/health-book.png", width=100)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Make Prediction", "📊 Model Info", "📈 Analytics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This AI assistant uses multimodal machine learning to analyze "
            "medical history and wearable device data for health predictions."
        )
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Make Prediction":
        show_prediction_page(fusion_model, config)
    elif page == "📊 Model Info":
        show_model_info_page(fusion_model, config)
    elif page == "📈 Analytics":
        show_analytics_page(config)


def show_home_page():
    """Display home page."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Advanced AI</h3>
            <p>Uses BERT for medical text analysis and LSTM for time-series processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Multimodal</h3>
            <p>Combines medical records and wearable device data for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Explainable</h3>
            <p>Provides SHAP-based explanations for model predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🚀 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Data Processing
        - ✅ Medical history text analysis with BERT
        - ✅ Wearable device time-series processing
        - ✅ Automated data validation and cleaning
        - ✅ Smart feature extraction
        """)
    
    with col2:
        st.markdown("""
        ### Model Capabilities
        - ✅ Binary disease risk prediction
        - ✅ Real-time inference
        - ✅ Model explainability with SHAP
        - ✅ Comprehensive performance metrics
        """)
    
    st.markdown("---")
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Navigate** to the "Make Prediction" page
    2. **Enter** patient medical history and wearable data
    3. **Click** "Predict" to get health risk assessment
    4. **Review** the prediction and confidence score
    """)


def show_prediction_page(fusion_model, config):
    """Display prediction page."""
    st.markdown("## 🔮 Make Health Prediction")
    st.markdown("Enter patient information to get a health risk assessment.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Medical History")
        
        age = st.slider("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        conditions = st.multiselect(
            "Medical Conditions",
            ["Hypertension", "Diabetes", "Asthma", "Arthritis", "None"],
            default=["None"]
        )
        
        symptoms = st.multiselect(
            "Current Symptoms",
            ["Chest Pain", "Shortness of Breath", "Fatigue", "Dizziness", 
             "Headache", "Nausea", "Joint Pain", "No Symptoms"],
            default=["No Symptoms"]
        )
        
        medications = st.multiselect(
            "Current Medications",
            ["Aspirin", "Metformin", "Lisinopril", "Albuterol", "None"],
            default=["None"]
        )
    
    with col2:
        st.markdown("### ⌚ Wearable Device Data")
        st.caption("Average values from the past week")
        
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 72)
        steps = st.number_input("Daily Steps", 0, 30000, 7500)
        sleep_hours = st.number_input("Sleep Hours", 0.0, 14.0, 7.5, step=0.5)
        calories = st.number_input("Daily Calories", 1000, 5000, 2000)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Health Risk", use_container_width=True):
        with st.spinner("Analyzing data..."):
            try:
                # Create medical history text
                history_parts = [f"{age} year old {gender.lower()} patient"]
                if conditions and conditions != ["None"]:
                    history_parts.append(f"with history of {', '.join(conditions).lower()}")
                if symptoms and symptoms != ["No Symptoms"]:
                    history_parts.append(f"presenting with {', '.join(symptoms).lower()}")
                if medications and medications != ["None"]:
                    history_parts.append(f"currently taking {', '.join(medications).lower()}")
                
                medical_history = ". ".join(history_parts) + "."
                
                # Process medical text
                medical_processor = MedicalDataProcessor(batch_size=1)
                text_embedding = medical_processor.preprocess_medical_text([medical_history])
                
                # Process time-series data
                ts_df = pd.DataFrame({
                    'heart_rate': [heart_rate],
                    'steps': [steps],
                    'sleep_hours': [sleep_hours],
                    'calories': [calories]
                })
                
                ts_processor = TimeSeriesProcessor(
                    feature_columns=config['columns']['time_series']
                )
                ts_data = ts_processor.preprocess_time_series(ts_df, fit=False)
                
                # Make prediction
                X = [text_embedding, ts_data]
                prediction = fusion_model.predict(X)[0][0]
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = "HIGH" if prediction > 0.5 else "LOW"
                    risk_color = "🔴" if prediction > 0.5 else "🟢"
                    st.metric("Risk Level", f"{risk_color} {risk_level}")
                
                with col2:
                    st.metric("Confidence", f"{prediction*100:.1f}%")
                
                with col3:
                    status = "At Risk" if prediction > 0.5 else "Healthy"
                    st.metric("Status", status)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Health Risk Score", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 30], 'color': '#90EE90'},
                            {'range': [30, 70], 'color': '#FFD700'},
                            {'range': [70, 100], 'color': '#FF6B6B'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if prediction > 0.7:
                    st.error("⚠️ High risk detected. Please consult a healthcare professional immediately.")
                elif prediction > 0.5:
                    st.warning("⚠️ Moderate risk detected. Consider scheduling a check-up.")
                else:
                    st.success("✅ Low risk. Continue maintaining healthy habits!")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def show_model_info_page(fusion_model, config):
    """Display model information page."""
    st.markdown("## 📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Architecture")
        st.info("""
        **Multimodal Fusion Model**
        - Text Branch: BERT embeddings → Dense layers
        - Time-Series Branch: Conv1D → LSTM → Dense
        - Fusion: Concatenation → Dense layers → Output
        """)
        
        if fusion_model.model:
            st.markdown("### Model Parameters")
            st.write(f"Total Parameters: {fusion_model.model.count_params():,}")
    
    with col2:
        st.markdown("### Configuration")
        st.json({
            "Model": {
                "Text Dense Units": config['model']['text_dense_units'],
                "TS Conv Filters": config['model']['ts_conv_filters'],
                "TS LSTM Units": config['model']['ts_lstm_units'],
                "Dropout Rate": config['model']['dropout_rate']
            },
            "Training": {
                "Epochs": config['training']['epochs'],
                "Batch Size": config['training']['batch_size'],
                "Learning Rate": config['model']['learning_rate']
            }
        })
    
    # Load metrics if available
    metrics_path = os.path.join(config['data']['output_dir'], 'evaluation', 'metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")


def show_analytics_page(config):
    """Display analytics page."""
    st.markdown("## 📈 Data Analytics")
    
    # Load data if available
    medical_path = config['data']['medical_history']
    wearable_path = config['data']['wearable_data']
    
    if not os.path.exists(medical_path) or not os.path.exists(wearable_path):
        st.warning("Data files not found. Generate data first using `python data_generator.py`")
        return
    
    medical_df = pd.read_csv(medical_path)
    wearable_df = pd.read_csv(wearable_path)
    
    # Merge data
    combined_df = pd.merge(medical_df, wearable_df, on='patient_id', how='inner')
    
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", len(combined_df))
    with col2:
        disease_count = combined_df['disease_label'].sum()
        st.metric("At Risk", disease_count)
    with col3:
        healthy_count = len(combined_df) - disease_count
        st.metric("Healthy", healthy_count)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(combined_df, x='age', color='disease_label',
                          title='Age Distribution by Health Status',
                          labels={'disease_label': 'At Risk'},
                          color_discrete_map={0: '#90EE90', 1: '#FF6B6B'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Heart rate distribution
        fig = px.box(combined_df, x='disease_label', y='heart_rate',
                    title='Heart Rate by Health Status',
                    labels={'disease_label': 'At Risk', 'heart_rate': 'Heart Rate (bpm)'},
                    color='disease_label',
                    color_discrete_map={0: '#90EE90', 1: '#FF6B6B'})
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Steps distribution
        fig = px.box(combined_df, x='disease_label', y='steps',
                    title='Daily Steps by Health Status',
                    labels={'disease_label': 'At Risk', 'steps': 'Daily Steps'},
                    color='disease_label',
                    color_discrete_map={0: '#90EE90', 1: '#FF6B6B'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sleep hours distribution
        fig = px.box(combined_df, x='disease_label', y='sleep_hours',
                    title='Sleep Hours by Health Status',
                    labels={'disease_label': 'At Risk', 'sleep_hours': 'Sleep Hours'},
                    color='disease_label',
                    color_discrete_map={0: '#90EE90', 1: '#FF6B6B'})
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
