# Models Directory

This directory contains trained models and related files.

## Supported service artifacts

- `../best_car_model.keras`: preferred Keras 3 model.
- `../car_classification_model.h5`: legacy Keras HDF5 fallback.

## Model Information

- **Architecture**: EfficientNetV2-S + GAP + dropout + 196-way softmax
- **Input Shape**: (224, 224, 3) RGB in `[0, 1]`
- **Output Classes**: 196 (Stanford Cars Dataset)
- **Training**: Transfer learning, then partial backbone fine-tune
- **Held-out test**: 81.88% top-1 / 95.54% top-5 (run `mild-aug-v1`)
- **Integrity**: root `model_manifest.json` authenticates the selected local
  artifact and exact class mapping; model weights remain gitignored and are not
  distributed by Git

## Usage

Without a manifest, the API service discovers supported artifacts from the
repository root in this legacy order:

1. Keras v3 (`best_car_model.keras`)
2. HDF5 (`car_classification_model.h5`)

When `model_manifest.json` is present, its declared artifact is authoritative
and the loader does not fall back to another file.

## Note

Keras 3 cannot load a TensorFlow SavedModel directory with `load_model()`. A
legacy `car_classification_savedmodel/` directory may still be useful for
TensorFlow Serving, but it is not a supported artifact for this API. Re-export
it to `.keras` before local or Docker use.

Model files are generated after running the training notebook and are not
included in the repository due to size constraints.
