# Models Directory

This directory contains trained models and related files.

## Files

- `car_classification_savedmodel/`: TensorFlow SavedModel format (preferred for deployment)
- `car_classification_model.h5`: Keras HDF5 model file (fallback)

## Model Information

- **Architecture**: ResNet50 + Custom Classification Head
- **Input Shape**: (224, 224, 3)
- **Output Classes**: 196 (Stanford Cars Dataset)
- **Training**: Transfer Learning + Fine-tuning

## Usage

The API service will automatically load models from this directory. The loading priority is:

1. SavedModel format (`car_classification_savedmodel/`)
2. HDF5 format (`car_classification_model.h5`)

## Note

Model files are generated after running the training notebook and are not included in the repository due to size constraints.
