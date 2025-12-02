# 🚀 Quick Start Guide

Welcome to the Personalized AI Health Assistant! This guide will get you up and running in minutes.

## 📋 Prerequisites

- Python 3.8 or higher installed
- pip package manager
- 4GB+ RAM recommended

## ⚡ Quick Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This will download ~2GB of dependencies including TensorFlow and BERT models. It may take 5-10 minutes depending on your internet connection.

### Step 2: Generate Sample Data

```bash
python data_generator.py
```

This creates:
- `data/medical_history.csv` - 1000 synthetic patient records
- `data/wearable_data.csv` - Corresponding wearable device data
- `data/combined_data.csv` - Merged dataset

### Step 3: Train the Model

```bash
python main.py
```

**Training time**: 
- CPU: ~30-45 minutes
- GPU: ~10-15 minutes

The script will:
- ✅ Process medical text with BERT
- ✅ Process time-series data
- ✅ Train the multimodal fusion model
- ✅ Evaluate and save results

### Step 4: Launch Web Interface

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start making predictions!

---

## 🎯 What You Can Do

### 1. Make Health Predictions

Navigate to the "Make Prediction" page and:
1. Enter patient information (age, gender, conditions)
2. Add wearable device data (heart rate, steps, sleep)
3. Click "Predict" to get instant risk assessment

### 2. View Model Performance

Check the "Model Info" page to see:
- Model architecture details
- Training configuration
- Performance metrics (accuracy, precision, recall, F1)

### 3. Explore Data Analytics

Visit the "Analytics" page for:
- Dataset statistics
- Distribution visualizations
- Health pattern insights

---

## 🔧 Customization

### Adjust Model Parameters

Edit `config.yaml` to customize:

```yaml
model:
  text_dense_units: 128      # Increase for more capacity
  dropout_rate: 0.3          # Adjust for regularization
  learning_rate: 0.001       # Tune for convergence

training:
  epochs: 50                 # More epochs = better fit
  batch_size: 32             # Adjust based on RAM
```

### Use Your Own Data

Replace the CSV files in `data/` with your own:

**Medical History Format:**
```csv
patient_id,age,gender,medical_history,disease_label
P00001,45,Male,"Patient history text...",1
```

**Wearable Data Format:**
```csv
patient_id,heart_rate,steps,sleep_hours,calories
P00001,75.2,8500,7.5,2100
```

---

## 📊 Understanding the Output

### Training Output

The training pipeline creates:

```
outputs/
├── training_history.png      # Loss and accuracy curves
├── training.log               # Detailed training logs
└── evaluation/
    ├── confusion_matrix.png   # Classification matrix
    ├── roc_curve.png          # ROC curve
    └── metrics.json           # Performance metrics

models/
├── best_model.keras           # Best model during training
├── final_model.keras          # Final trained model
└── model_metadata.json        # Model information
```

### Prediction Output

When you make a prediction, you get:
- **Risk Level**: HIGH or LOW
- **Confidence**: Probability score (0-100%)
- **Status**: At Risk or Healthy
- **Recommendations**: Actionable health advice

---

## 🐛 Troubleshooting

### Issue: "BERT model download failed"

**Solution**: 
```bash
# Set cache directory
export TRANSFORMERS_CACHE=./transformers_cache
python main.py
```

### Issue: "Out of memory"

**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # or even 8
```

### Issue: "Training is too slow"

**Solutions**:
1. Reduce epochs: `epochs: 20`
2. Use smaller dataset: Modify `data_generator.py` to generate fewer samples
3. Skip SHAP explanations: Comment out Step 7 in `main.py`

### Issue: "Module not found"

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

---

## 🎓 Next Steps

### Improve Model Performance

1. **Collect more data**: More samples = better generalization
2. **Feature engineering**: Add more wearable metrics
3. **Hyperparameter tuning**: Experiment with config values
4. **Add image data**: Integrate medical imaging

### Deploy to Production

1. **Containerize**: Create Docker image
2. **API**: Convert to FastAPI endpoint
3. **Cloud**: Deploy to AWS/GCP/Azure
4. **Monitor**: Add logging and monitoring

### Extend Functionality

1. **Multi-class**: Predict specific diseases
2. **Regression**: Predict continuous health scores
3. **Time-series forecasting**: Predict future health trends
4. **Personalized recommendations**: Generate custom health plans

---

## 💡 Tips for Best Results

1. **Data Quality**: Clean, accurate data is crucial
2. **Balanced Dataset**: Equal samples of healthy/at-risk patients
3. **Regular Updates**: Retrain with new data periodically
4. **Validation**: Always validate predictions with medical professionals
5. **Privacy**: Ensure HIPAA/GDPR compliance for real patient data

---

## 📚 Learn More

- **BERT**: [Hugging Face Documentation](https://huggingface.co/docs/transformers)
- **LSTM**: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **SHAP**: [SHAP Documentation](https://shap.readthedocs.io/)
- **Streamlit**: [Streamlit Docs](https://docs.streamlit.io/)

---

## ❓ Need Help?

- 📖 Check the [full README](README.markdown)
- 🐛 [Open an issue](https://github.com/ValteruGowtham/Personalized-AI-Health-Assistant-Using-Multimodal-Machine-Learning/issues)
- 💬 Review the code comments
- 🧪 Run tests: `pytest test_health_assistant.py -v`

---

<div align="center">
  <strong>Happy Predicting! 🎉</strong>
</div>
