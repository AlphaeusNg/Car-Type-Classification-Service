# Stanford Cars Dataset

This directory contains scripts and utilities for downloading and preparing the Stanford Cars Dataset.

## Dataset Information

- **Name**: Stanford Cars Dataset
- **Classes**: 196 (car make, model, year combinations)
- **Total Images**: 16,185
- **Training Images**: 8,144
- **Test Images**: 8,041
- **Image Size**: ~360x240 pixels (variable)

## Download Instructions

### Option 1: Kaggle (Recommended)

1. Install Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle credentials (create account and download API key from kaggle.com/account)

3. Download dataset:
   ```bash
   kaggle datasets download -d jessicali9530/stanford-cars-dataset
   ```

### Option 2: Manual Download

Visit the [Kaggle Dataset Page](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) and download manually.

### Option 3: Original Source

The original dataset can be found at the [Stanford Cars Dataset page](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), though the download links may be broken.

## Data Structure

After downloading and extracting, the expected structure is:

```
data/
├── cars_train/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── cars_test/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── cars_train_annos.mat
├── cars_test_annos.mat
└── cars_meta.mat
```

## Notes

- The notebook in this project uses synthetic data for demonstration
- For production use, replace the synthetic data generation with actual dataset loading
- Ensure proper licensing compliance when using the dataset
