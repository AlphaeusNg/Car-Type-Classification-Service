#!/usr/bin/env python3
"""
Car Type Classification - Prediction Example
Example code showing how to use the saved model for predictions.
"""

import tensorflow as tf
import numpy as np
import json
from PIL import Image

def predict_car_type(image_path, model_path='best_car_model.keras', 
                    mapping_path='class_mapping.json'):
    """
    Predict car type from image file

    Args:
        image_path: Path to the car image
        model_path: Path to the trained model
        mapping_path: Path to class mapping JSON

    Returns:
        Dictionary with prediction results
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load class mappings
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # Load and preprocess image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Get class name
    predicted_class = class_mapping['index_to_class'][str(predicted_class_idx)]

    # Get top 5 predictions
    top5_indices = np.argsort(predictions[0])[-5:][::-1]
    top5_predictions = [
        {
            'class': class_mapping['index_to_class'][str(idx)],
            'confidence': float(predictions[0][idx])
        }
        for idx in top5_indices
    ]

    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'class_index': int(predicted_class_idx),
        'top5_predictions': top5_predictions
    }

# Example usage:
if __name__ == "__main__":
    # Test with a sample image (replace with your image path)
    image_path = "data/test/BMW M3 Coupe 2012/00001.jpg"  # Example path

    try:
        result = predict_car_type(image_path)
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nTop 5 predictions:")
        for i, pred in enumerate(result['top5_predictions'], 1):
            print(f"  {i}. {pred['class']} ({pred['confidence']:.2%})")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the image path exists and model files are available")
