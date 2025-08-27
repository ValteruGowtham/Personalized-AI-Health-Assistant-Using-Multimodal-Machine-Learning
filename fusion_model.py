from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv1D, LSTM
import shap
import numpy as np

def build_fusion_model(text_shape, ts_shape):
    """
    Build multimodal fusion model.
    Args:
        text_shape (tuple): Shape of BERT embeddings.
        ts_shape (tuple): Shape of time-series data.
    Returns:
        Model: Compiled Keras model.
    """
    # Text branch (BERT embeddings)
    text_input = Input(shape=text_shape, name='text_input')
    text_dense = Dense(64, activation='relu')(text_input)

    # Time-series branch
    ts_input = Input(shape=ts_shape, name='ts_input')
    ts_conv = Conv1D(32, kernel_size=3, activation='relu')(ts_input)
    ts_lstm = LSTM(32)(ts_conv)
    ts_dense = Dense(32, activation='relu')(ts_lstm)

    # Fusion
    concat = Concatenate()([text_dense, ts_dense])
    dense = Dense(64, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)  # Binary disease prediction

    # Compile model
    model = Model(inputs=[text_input, ts_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, text_data, ts_data, labels):
    """
    Train the fusion model.
    Args:
        model: Compiled Keras model.
        text_data (np.array): BERT embeddings.
        ts_data (np.array): Time-series data.
        labels (np.array): Target labels.
    """
    model.fit([text_data, ts_data], labels, epochs=10, batch_size=32, validation_split=0.2)

def explain_model(model, text_data, ts_data):
    """
    Explain predictions using SHAP.
    Args:
        model: Trained Keras model.
        text_data (np.array): BERT embeddings.
        ts_data (np.array): Time-series data.
    """
    explainer = shap.KernelExplainer(model.predict, [text_data[:100], ts_data[:100]])
    shap_values = explainer.shap_values([text_data[:100], ts_data[:100]])
    shap.summary_plot(shap_values, feature_names=['text', 'time_series'])