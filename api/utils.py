"""
Utility functions for the Car Type Classification API
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os
from typing import Dict, Any

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def load_model() -> tf.keras.Model:
    """
    Load the trained car classification model
    
    Returns:
        Loaded Keras model
    """
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Try different model file formats and locations
        model_paths = [
            os.path.join(project_root, "best_car_model.keras"),  # TF 2.19 format
            os.path.join(project_root, "car_classification_model.h5"),  # Legacy format
            os.path.join(project_root, "models", "car_classification_savedmodel"),  # SavedModel format
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully! Input shape: {model.input_shape}")
                return model
        
        # If no model found, raise error with helpful message
        available_files = os.listdir(project_root)
        model_files = [f for f in available_files if f.endswith(('.h5', '.keras')) or 'model' in f.lower()]
        
        raise FileNotFoundError(
            f"No model file found. Checked paths: {model_paths}\n"
            f"Available model-related files in {project_root}: {model_files}\n"
            f"Please ensure you have a trained model file."
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def load_class_mapping() -> Dict[str, Any]:
    """
    Load class mapping for car types
    
    Returns:
        Dictionary containing class mappings
    """
    try:
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        class_mapping_path = os.path.join(project_root, 'class_mapping.json')
        
        if not os.path.exists(class_mapping_path):
            print("Class mapping file not found, creating a sample one...")
            create_sample_class_mapping()
        
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        print(f"Class mapping loaded with {class_mapping.get('num_classes', 'unknown')} classes")
        return class_mapping
        
    except Exception as e:
        raise RuntimeError(f"Failed to load class mapping: {str(e)}")

def create_sample_class_mapping():
    """
    Create a sample class mapping file if it doesn't exist
    This is for demonstration purposes
    """
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    class_mapping_path = os.path.join(project_root, 'class_mapping.json')
    
    if not os.path.exists(class_mapping_path):
        # Create a sample class mapping
        car_classes = [
            "Acura Integra Type R 2001",
            "Acura RL Sedan 2012",
            "Acura TL Sedan 2012",
            "Acura TL Type-S 2008",
            "Acura TSX Sedan 2012",
            "Acura ZDX Hatchback 2012",
            "Aston Martin V8 Vantage Convertible 2012",
            "Aston Martin V8 Vantage Coupe 2012",
            "Aston Martin Virage Convertible 2012",
            "Aston Martin Virage Coupe 2012",
            "BMW M3 Coupe 2012",
            "BMW M5 Sedan 2010",
            "BMW X3 SUV 2012",
            "Ford F-150 Regular Cab 2012",
            "Toyota Camry Sedan 2012"
        ]
        
        # Extend to 196 classes for Stanford Cars Dataset
        all_classes = car_classes + [f"Car Class {i+16}" for i in range(181)]
        
        class_mapping = {
            'class_to_index': {cls: idx for idx, cls in enumerate(all_classes)},
            'index_to_class': {str(idx): cls for idx, cls in enumerate(all_classes)},
            'num_classes': len(all_classes)
        }
        
        with open(class_mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"Sample class mapping created at: {class_mapping_path}")
    else:
        print(f"Class mapping already exists at: {class_mapping_path}")
