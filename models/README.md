# Models Directory

This directory contains trained models and related files.

## Supported service artifacts

- `../best_car_model.keras`: preferred Keras 3 model.
- `../car_classification_model.h5`: legacy Keras HDF5 fallback.

## Model Information

- **Architecture**: ResNet50 + Custom Classification Head
- **Input Shape**: (224, 224, 3)
- **Output Classes**: 196 (Stanford Cars Dataset)
- **Training**: Transfer Learning + Fine-tuning

## Usage

The API service automatically loads supported artifacts from the repository
root in this order:

1. Keras v3 (`best_car_model.keras`)
2. HDF5 (`car_classification_model.h5`)

## Note

Keras 3 cannot load a TensorFlow SavedModel directory with `load_model()`. A
legacy `car_classification_savedmodel/` directory may still be useful for
TensorFlow Serving, but it is not a supported artifact for this API. Re-export
it to `.keras` before local or Docker use.

Model files are generated after running the training notebook and are not
included in the repository due to size constraints.
