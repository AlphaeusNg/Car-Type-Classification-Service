"""Dependency-free model artifact discovery shared by launch and inference."""

from pathlib import Path


MODEL_CANDIDATES = (
    Path("best_car_model.keras"),
    Path("car_classification_model.h5"),
)


def find_model_artifact(root=Path(".")):
    """Return the first Keras 3 artifact supported by the API loader."""
    root = Path(root)
    for relative_path in MODEL_CANDIDATES:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None
