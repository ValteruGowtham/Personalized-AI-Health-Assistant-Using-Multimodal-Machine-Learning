# 🩺 Personalized AI Health Assistant Using Multimodal Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **A cutting-edge AI health assistant** that leverages multimodal machine learning to deliver personalized health insights by integrating medical records, time-series data from wearable devices, and advanced deep learning models.

![Health Assistant Banner](https://img.shields.io/badge/AI-Health%20Assistant-blueviolet?style=for-the-badge)

---

## 📖 Overview

This project combines diverse data types to provide tailored health risk assessments and diagnostics, powered by state-of-the-art machine learning models including BERT for medical text analysis and LSTM networks for time-series processing.

### ✨ Key Features

- 🧠 **Advanced NLP**: BERT-based medical text understanding
- 📊 **Time-Series Analysis**: LSTM networks for wearable device data
- 🔄 **Multimodal Fusion**: Intelligent combination of multiple data sources
- 🎯 **High Accuracy**: Comprehensive model with dropout and batch normalization
- 📈 **Model Explainability**: SHAP-based prediction explanations
- 🌐 **Web Interface**: Beautiful Streamlit dashboard for predictions
- ✅ **Production Ready**: Error handling, logging, and model persistence

---

## 📂 Project Structure

```
medical-ass/
├── 📄 main.py                    # Main training pipeline
├── 📄 app.py                     # Streamlit web interface
├── 📄 config.yaml                # Configuration file
├── 📄 utils.py                   # Utility functions
├── 📄 data_generator.py          # Synthetic data generation
├── 🧠 medical_data.py            # Medical text processing (BERT)
├── ⏳ time_series.py             # Time-series data processing
├── 🖼️ image_data.py              # Medical image processing
├── 🔮 fusion_model.py            # Multimodal fusion model
├── 🧪 test_health_assistant.py  # Unit tests
├── 📋 requirements.txt           # Python dependencies
├── 📁 data/                      # Data directory
├── 📁 models/                    # Saved models
└── 📁 outputs/                   # Results and visualizations
```

---

## ⚙️ Setup

### Prerequisites

- 🐍 Python 3.8 or higher
- 💾 At least 4GB RAM (8GB recommended for BERT)
- 🎮 GPU optional but recommended for faster training

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ValteruGowtham/Personalized-AI-Health-Assistant-Using-Multimodal-Machine-Learning.git
   cd Personalized-AI-Health-Assistant-Using-Multimodal-Machine-Learning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data:**
   ```bash
   python data_generator.py
   ```

---

## 🚀 Usage

### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

This will:
- ✅ Load and process medical and wearable data
- ✅ Split data into train/validation/test sets
- ✅ Build and train the multimodal fusion model
- ✅ Evaluate performance with comprehensive metrics
- ✅ Generate visualizations and SHAP explanations
- ✅ Save the trained model and metadata

### Running the Web Interface

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

The web interface provides:
- 🔮 **Prediction Page**: Enter patient data and get instant health risk assessments
- 📊 **Model Info**: View model architecture and performance metrics
- 📈 **Analytics**: Explore data distributions and insights
- 🏠 **Home**: Overview of features and capabilities

### Running Tests

Execute unit tests:

```bash
pytest test_health_assistant.py -v
```

---

## 📊 Model Architecture

The system uses a sophisticated multimodal fusion architecture:

### Text Branch (Medical History)
```
Input → BERT Embeddings (768d) → Dense(128) → BatchNorm → Dropout → Dense(64)
```

### Time-Series Branch (Wearable Data)
```
Input → Conv1D(64) → BatchNorm → LSTM(64) → Dropout → GlobalAvgPool → Dense(64)
```

### Fusion Layer
```
Concatenate → Dense(128) → BatchNorm → Dropout → Dense(64) → Dropout → Output(1)
```

**Total Parameters**: ~110M (BERT) + ~50K (custom layers)

---

## 📈 Performance Metrics

The model is evaluated using:
- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ ROC-AUC
- ✅ Confusion Matrix
- ✅ ROC Curve

All metrics and visualizations are automatically saved to the `outputs/` directory.

---

## 🔧 Configuration

Customize the model and training parameters in `config.yaml`:

```yaml
model:
  text_dense_units: 128
  ts_conv_filters: 64
  ts_lstm_units: 64
  dropout_rate: 0.3
  learning_rate: 0.001

training:
  epochs: 50
  batch_size: 32
  validation_split: 0.15
  test_split: 0.15
  early_stopping_patience: 10
```

---

## 📋 Data Format

### Medical History CSV
```csv
patient_id,age,gender,medical_history,disease_label
P00001,45,Male,"45 year old male patient with history of hypertension...",1
```

### Wearable Data CSV
```csv
patient_id,heart_rate,steps,sleep_hours,calories
P00001,75.2,8500,7.5,2100
```

---

## 🎯 Use Cases

- 🏥 **Clinical Decision Support**: Assist healthcare providers with risk assessment
- 📱 **Personal Health Monitoring**: Track and predict health trends
- 🔬 **Medical Research**: Analyze patterns in multimodal health data
- 💊 **Preventive Care**: Early detection of health risks

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. 🍴 Fork the repository
2. 🌱 Create a feature branch: `git checkout -b feature-name`
3. 💾 Commit changes: `git commit -m 'Add feature'`
4. 🚀 Push to branch: `git push origin feature-name`
5. 📬 Open a pull request

---

## 🐛 Troubleshooting

### Common Issues

**BERT model download fails:**
```bash
# Set HuggingFace cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

**Out of memory errors:**
- Reduce batch size in `config.yaml`
- Use CPU instead of GPU for BERT
- Process data in smaller chunks

**Model training is slow:**
- Enable GPU acceleration
- Reduce number of epochs
- Use smaller BERT model (bert-small)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🤗 Hugging Face for BERT models
- 🧠 TensorFlow team for the deep learning framework
- 📊 SHAP library for model explainability
- 🎨 Streamlit for the web interface framework

---

## 📧 Contact

**Valteru Gowtham**

- GitHub: [@ValteruGowtham](https://github.com/ValteruGowtham)
- Project Link: [Personalized AI Health Assistant](https://github.com/ValteruGowtham/Personalized-AI-Health-Assistant-Using-Multimodal-Machine-Learning)

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐!

---

<div align="center">
  <strong>Made with ❤️ for better healthcare through AI</strong>
</div>