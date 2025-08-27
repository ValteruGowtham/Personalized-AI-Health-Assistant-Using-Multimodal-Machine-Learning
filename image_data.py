import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single medical image.
    Args:
        image_path (str): Path to image file.
        target_size (tuple): Desired image size.
    Returns:
        np.array: Preprocessed image array.
    """
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0  # Normalize
    return img

def load_image_data(image_paths):
    """
    Load and preprocess multiple images.
    Args:
        image_paths (list): List of image file paths.
    Returns:
        np.array: Preprocessed image data.
    """
    images = [preprocess_image(path) for path in image_paths]
    return np.array(images)