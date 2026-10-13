#!/usr/bin/env python3
"""
Car Type Classification - Prediction Example
Example code showing how to use the saved model for predictions.
"""

from pathlib import Path

from api.utils import (
    decode_predictions,
    load_class_mapping,
    load_model,
    preprocess_image,
    validate_runtime_artifacts,
)


def predict_car_type(
    image_path,
    model_path=None,
    mapping_path=None,
    *,
    model_loader=None,
    top_k=5,
):
    """
    Predict car type from image file

    Args:
        image_path: Path to the car image
        model_path: Optional explicit model; defaults to the repository manifest
        mapping_path: Optional mapping; defaults to the repository class mapping
        model_loader: Optional Keras-compatible loader for testing
        top_k: Number of ranked candidates (default 5)

    Returns:
        Dictionary with prediction results
    """
    image_array = preprocess_image(Path(image_path).read_bytes())
    if model_loader is None:
        import tensorflow as tf

        model_loader = tf.keras.models.load_model

    project_root = Path(__file__).resolve().parent
    if model_path is None:
        model = load_model(project_root=project_root, model_loader=model_loader)
    else:
        from api.model_artifacts import verify_model_artifact

        selected = Path(model_path).resolve()
        verify_model_artifact(selected, selected.parent)
        model = model_loader(str(selected), compile=False)
    class_mapping = load_class_mapping(
        Path(mapping_path) if mapping_path is not None else project_root / "class_mapping.json"
    )
    validate_runtime_artifacts(model, class_mapping)
    result = decode_predictions(
        model.predict(image_array, verbose=0),
        class_mapping["index_to_class"],
        top_k=top_k,
    )
    result["class_index"] = class_mapping["class_to_index"][
        result["predicted_class"]
    ]
    return result


def main(argv=None):
    import argparse
    from contextlib import nullcontext, redirect_stdout
    import json
    import sys

    parser = argparse.ArgumentParser(description="Classify one local JPEG or PNG car image.")
    parser.add_argument("image", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "data/test/Acura TL Sedan 2012/000197.jpg")
    parser.add_argument("--model", type=Path, help="Explicit local model; otherwise use the repository manifest")
    parser.add_argument("--mapping", type=Path, help="Explicit class mapping JSON")
    parser.add_argument("--json", action="store_true", help="Write one JSON result to stdout; diagnostics to stderr")
    parser.add_argument("--top-k", type=int, choices=range(1, 197), default=5,
                        metavar="1..196", help="Number of ranked class scores (default: 5)")
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise FileNotFoundError(f"Image file does not exist: {args.image}")
        with redirect_stdout(sys.stderr) if args.json else nullcontext():
            result = predict_car_type(args.image, args.model, args.mapping, top_k=args.top_k)
        if args.json:
            payload = {
                "schema_version": 1,
                "status": "success",
                "predicted_class": result["predicted_class"],
                "class_index": result["class_index"],
                "confidence": result["confidence"],
                "top_predictions": result["top5_predictions"],
            }
            # Serialize fully before writing so failures cannot leave partial JSON.
            print(json.dumps(payload, allow_nan=False))
            return 0
        print(f"Predicted: {result['predicted_class']}")
        # A softmax score is not a calibrated probability of label correctness.
        print(f"Class score: {result['confidence']:.2%}")
        print(f"\nTop {args.top_k} class scores:")
        for i, pred in enumerate(result["top5_predictions"], 1):
            print(f"  {i}. {pred['class']} ({pred['confidence']:.2%})")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
