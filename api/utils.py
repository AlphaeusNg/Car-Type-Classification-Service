"""
Car Type Classification API Utilities
Helper functions for image preprocessing and model loading
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import json
import io
from typing import TYPE_CHECKING, Dict, Any
from pathlib import Path

if TYPE_CHECKING:
    import tensorflow as tf


def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Preprocessed image array (1, 224, 224, 3) normalized to [0,1]
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

def load_model(project_root: Path | None = None, model_loader=None) -> tf.keras.Model:
    """
    Load the trained car classification model
    
    Returns:
        Loaded Keras model
        
    Raises:
        FileNotFoundError: If no model file is found
        RuntimeError: If model loading fails
    """
    if model_loader is None:
        import tensorflow as tf

        model_loader = tf.keras.models.load_model

    if project_root is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
    else:
        project_root = Path(project_root)
    
    # Try different model formats in order of preference
    model_paths = [
        project_root / "best_car_model.keras",     # TF 2.19+ format (preferred)
        project_root / "car_classification_model.h5",  # Legacy format
        project_root / "models" / "car_classification_savedmodel",  # SavedModel
    ]
    
    load_failures = []
    for model_path in model_paths:
        if model_path.exists():
            try:
                print(f"🔄 Loading model from: {model_path}")
                model = model_loader(str(model_path))
                print(f"✅ Model loaded! Input shape: {model.input_shape}")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load {model_path}: {e}")
                load_failures.append(f"{model_path}: {type(e).__name__}: {e}")
                continue

    if load_failures:
        raise RuntimeError(
            "❌ Model artifacts were found but none could be loaded:\n"
            + "\n".join(load_failures)
        )

    # No model found - provide helpful error
    available_files = [f.name for f in project_root.iterdir() 
                      if f.suffix in ['.h5', '.keras'] or 'model' in f.name.lower()]
    
    raise FileNotFoundError(
        f"❌ No model file found! Checked: {[str(p) for p in model_paths]}\n"
        f"Available files: {available_files}\n"
        f"Please train the model first using: jupyter notebook model_training.ipynb"
    )

def validate_class_mapping(class_mapping: Any) -> Dict[str, Any]:
    """Validate the minimum persisted mapping structure."""
    required_keys = ["index_to_class", "class_to_index"]
    if not isinstance(class_mapping, dict) or not all(
        key in class_mapping for key in required_keys
    ):
        raise ValueError(f"Invalid mapping format. Required keys: {required_keys}")
    return class_mapping


def load_class_mapping(mapping_path: Path | None = None) -> Dict[str, Any]:
    """
    Load class mapping for car types
    
    Returns:
        Dictionary with 'index_to_class' and 'class_to_index' mappings
        
    Raises:
        FileNotFoundError: If class mapping file not found
        ValueError: If class mapping format is invalid
    """
    # Get project root directory  
    if mapping_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        mapping_path = project_root / "class_mapping.json"
    else:
        mapping_path = Path(mapping_path)
    
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"❌ Class mapping not found: {mapping_path}\n"
            f"Please train the model first to generate class mappings."
        )
    
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)

        validate_class_mapping(class_mapping)
        print(f"✅ Class mapping loaded: {len(class_mapping['index_to_class'])} classes")
        return class_mapping

    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON in class mapping: {e}")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load class mapping: {e}")


def get_model_info(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Get model information for debugging/monitoring
    
    Args:
        model: Loaded Keras model
        
    Returns:
        Dictionary with model metadata
    """
    import tensorflow as tf

    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape, 
        "total_params": model.count_params(),
        "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        "layers": len(model.layers)
    }
