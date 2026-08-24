#!/usr/bin/env python3
"""
Car Type Classification - Prediction Example
Example code showing how to use the saved model for predictions.
"""

from pathlib import Path

from api.utils import (
    decode_predictions,
    load_class_mapping,
    preprocess_image,
    validate_runtime_artifacts,
)


def predict_car_type(
    image_path,
    model_path="best_car_model.keras",
    mapping_path="class_mapping.json",
    *,
    model_loader=None,
):
    """
    Predict car type from image file

    Args:
        image_path: Path to the car image
        model_path: Path to the trained model
        mapping_path: Path to class mapping JSON
        model_loader: Optional Keras-compatible loader for testing

    Returns:
        Dictionary with prediction results
    """
    if model_loader is None:
        import tensorflow as tf

        model_loader = tf.keras.models.load_model

    model = model_loader(str(Path(model_path)), compile=False)
    class_mapping = load_class_mapping(Path(mapping_path))
    validate_runtime_artifacts(model, class_mapping)
    image_array = preprocess_image(Path(image_path).read_bytes())
    result = decode_predictions(
        model.predict(image_array, verbose=0),
        class_mapping["index_to_class"],
    )
    result["class_index"] = class_mapping["class_to_index"][
        result["predicted_class"]
    ]
    return result


# Example usage:
if __name__ == "__main__":
    # Test with a sample image (replace with your image path)
    image_path = "data/test/Acura TL Sedan 2012/000197.jpg"  # Example path

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
