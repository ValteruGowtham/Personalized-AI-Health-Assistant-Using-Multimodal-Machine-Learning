from medical_data import load_medical_data
from time_series import load_time_series_data
from image_data import load_image_data
from fusion_model import build_fusion_model, train_model, explain_model

text_data = load_medical_data('medical_history.csv')
ts_data = load_time_series_data('wearable_data.csv')
# image_data = load_image_data(image_paths)  # Uncomment if using images
model = build_fusion_model(text_shape=(768,), ts_shape=ts_data.shape[1:])
train_model(model, text_data, ts_data, labels)  # Provide labels
explain_model(model, text_data, ts_data)