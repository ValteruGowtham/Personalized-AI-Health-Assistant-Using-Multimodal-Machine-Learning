"""
Medical image preprocessing module.
Handles medical imaging data (X-rays, MRIs, CT scans, etc.)
"""
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging
from typing import List, Tuple
import os

logger = logging.getLogger(__name__)


class ImageDataProcessor:
    """Process medical image data."""
    
    def __init__(self, target_size=(224, 224), use_pretrained=False):
        """
        Initialize image processor.
        
        Args:
            target_size (tuple): Target image size (height, width)
            use_pretrained (bool): Whether to use pretrained ResNet50 for feature extraction
        """
        self.target_size = target_size
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            logger.info("Loading pretrained ResNet50 for feature extraction...")
            self.feature_extractor = ResNet50(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(*target_size, 3)
            )
            logger.info("ResNet50 loaded successfully")
        else:
            self.feature_extractor = None
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single medical image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.array: Preprocessed image array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load image
            img = load_img(image_path, target_size=self.target_size)
            img_array = img_to_array(img)
            
            # Normalize
            if self.use_pretrained:
                # Use ResNet preprocessing
                img_array = preprocess_input(img_array)
            else:
                # Simple normalization
                img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features using pretrained model.
        
        Args:
            images (np.ndarray): Array of preprocessed images
            
        Returns:
            np.ndarray: Extracted features
        """
        if not self.use_pretrained:
            return images
        
        logger.info(f"Extracting features from {len(images)} images...")
        features = self.feature_extractor.predict(images, verbose=0)
        logger.info(f"Feature extraction complete. Shape: {features.shape}")
        
        return features
    
    def load_image_data(self, image_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess multiple images.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            np.array: Preprocessed image data
        """
        if not image_paths:
            raise ValueError("Empty image paths list provided")
        
        logger.info(f"Loading {len(image_paths)} images...")
        
        images = []
        failed_images = []
        
        for i, path in enumerate(image_paths):
            try:
                img = self.preprocess_image(path)
                images.append(img)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                failed_images.append(path)
        
        if not images:
            raise ValueError("No images were successfully loaded")
        
        if failed_images:
            logger.warning(f"Failed to load {len(failed_images)} images")
        
        # Convert to array
        images_array = np.array(images)
        logger.info(f"Loaded {len(images)} images. Shape: {images_array.shape}")
        
        # Extract features if using pretrained model
        if self.use_pretrained:
            images_array = self.extract_features(images_array)
        
        return images_array


def load_image_data(image_paths: List[str], 
                   target_size: Tuple[int, int] = (224, 224),
                   use_pretrained: bool = False) -> np.ndarray:
    """
    Convenience function to load and process image data.
    
    Args:
        image_paths (list): List of image file paths
        target_size (tuple): Target image size
        use_pretrained (bool): Whether to use pretrained model
        
    Returns:
        np.array: Preprocessed image data
    """
    processor = ImageDataProcessor(target_size=target_size, use_pretrained=use_pretrained)
    return processor.load_image_data(image_paths)