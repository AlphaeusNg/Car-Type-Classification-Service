# Stanford Cars Dataset

This directory contains the Stanford Cars Dataset used for training the car type classification model.

## Dataset Information

- **Name**: Stanford Cars Dataset (by Classes Folder)
- **Classes**: 196 (car make, model, year combinations)
- **Total Images**: 16,185
- **Training Images**: 8,144
- **Test Images**: 8,041
- **Source**: Pre-organized by classes for easy use with TensorFlow

## Download Instructions

The dataset is automatically downloaded when running the training notebook using KaggleHub:

```python
import kagglehub
download_path = kagglehub.dataset_download("cyizhuo/stanford-cars-by-classes-folder")
```

### Manual Download (Alternative)

If you prefer to download manually:

1. Visit the [Kaggle Dataset Page](https://www.kaggle.com/datasets/cyizhuo/stanford-cars-by-classes-folder)
2. Download and extract to your preferred location
3. Update the path in the training notebook

## Expected Data Structure

After the training notebook organizes the dataset, the structure will be:

```
data/
├── train/
│   ├── Acura Integra Type R 2001/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── Acura RL Sedan 2012/
│   └── ... (196 classes total)
└── test/
    ├── Acura Integra Type R 2001/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── Acura RL Sedan 2012/
    └── ... (196 classes total)
```

## Usage

1. **Automatic Download**: Simply run the training notebook - the dataset will be downloaded automatically
2. **Training**: The dataset is already organized by classes, making it ready for `tf.keras.utils.image_dataset_from_directory`
3. **Class Mapping**: Class names are automatically extracted from folder names

## Notes

- Dataset is automatically downloaded and organized by the training notebook
- No manual preprocessing required - ready to use with TensorFlow
- Total dataset size: ~1.8GB
- Ensure proper licensing compliance when using the dataset
