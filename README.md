# Car Type Classification Service

A deep learning image classification service for car types using a ResNet50-based model trained on the Stanford Cars Dataset. This project fulfills a complete model training in Jupyter notebook, REST API deployment, and Docker containerization.

## � Project Overview

This service classifies car images into 196 different car types (make, model, year combinations) using a transfer learning approach with ResNet50 backbone. The complete solution includes:

- **Deep Learning Model**: ResNet50 with transfer learning trained on Stanford Cars Dataset
- **REST API**: FastAPI service with `/predict` endpoint
- **Docker Support**: Complete containerization for deployment
- **Jupyter Training**: Comprehensive model training notebook

## 🏗️ Project Structure

```
├── model_training.ipynb           # Main training notebook (REQUIRED)
├── api/                          # API source code (REQUIRED)
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   └── utils.py                  # Preprocessing and model utilities
├── Dockerfile                    # Container configuration (REQUIRED)
├── requirements.txt              # Python dependencies (REQUIRED)
├── README.md                     # Documentation (REQUIRED)
├── run.py                        # Automated setup and deployment script
├── best_car_model.keras          # Trained model (generated)
├── class_mapping.json           # Class index mapping (generated)
└── data/                        # Stanford Cars dataset
    ├── train/                   # Training images
    └── test/                    # Test images
```

## � Quick Start

### Prerequisites
- Python 3.12+
- Docker (optional, for containerized deployment)
- 4GB+ RAM (8GB+ recommended for training)

### 1. Setup Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/AlphaeusNg/Car-Type-Classification-Service.git
cd Car-Type-Classification-Service

# Auto-setup environment and dependencies
python3 run.py --setup
```

### 2. Train the Model
```bash
# Launch Jupyter notebook
jupyter notebook model_training.ipynb

# Run all cells to:
# - Load and preprocess Stanford Cars dataset
# - Train ResNet50-based model
# - Generate evaluation metrics
# - Save model files (best_car_model.keras, class_mapping.json)
```

### 3. Run the API

#### Option A: Quick Start with run.py
```bash
python3 run.py --mode local    # Run locally on port 8000
python3 run.py --mode docker   # Run in Docker container
```

#### Option B: Manual Setup
```bash
# Local development
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker deployment
docker build -t car-classification-service .
docker run -p 8000:8000 car-classification-service
```

### Alternative Environment Setup

If you prefer manual setup:

**Python Virtual Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/WSL2
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Conda Environment:**
```bash
conda create -n car-classification python=3.12
conda activate car-classification
pip install -r requirements.txt
```

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Car Type Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@data/test/Acura TL Sedan 2012/000197.jpg"
```

**Response Format:**
```json
{
   "predicted_class": "Acura TL Sedan 2012",
   "confidence": 0.2989,
   "top5_predictions": [
      {
         "class": "Acura TL Sedan 2012",
         "confidence": 0.2989
      },
      {
         "class": "Chevrolet Malibu Hybrid Sedan 2010",
         "confidence": 0.2387
      },
      {
         "class": "Cadillac SRX SUV 2012",
         "confidence": 0.0466
      },
      {
         "class": "Audi A5 Coupe 2012",
         "confidence": 0.0443
      },
      {
         "class": "Hyundai Genesis Sedan 2012",
         "confidence": 0.0366
      }
   ],
   "status": "success"
}
```

**Interactive Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 Technical Details

### Model Architecture
- **Framework**: TensorFlow 2.19.0 with Keras
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Input Size**: 224×224×3 RGB images
- **Classes**: 196 Stanford Cars dataset classes
- **Training**: Transfer learning with fine-tuning
- **Data Split**: 80% training, 20% validation

### API Framework
- **Framework**: FastAPI 0.116.1
- **Server**: Uvicorn ASGI server
- **Image Processing**: Pillow for image preprocessing
- **Model Loading**: TensorFlow model loading utilities

### Deployment
- **Container**: Python 3.12-slim base image
- **Port**: 8000 (configurable)
- **Health Checks**: Built-in health monitoring
- **Error Handling**: Comprehensive error responses

## 📊 Model Performance

Performance metrics are generated in the notebook:
- **Training Accuracy**: ~95%+ 
- **Validation Accuracy**: ~50%+
- **Top-5 Accuracy**: ~75%+
- **Training Plots**: Loss and accuracy curves

## 🐛 Troubleshooting

### Common Issues

**Environment Setup:**
```bash
python3 run.py --setup  # Reset environment if issues occur
```

**Port Already in Use:**
```bash
python3 run.py --mode local --port 8080  # Use different port
```

**Missing Model Files:**
- Run `model_training.ipynb` notebook first to generate required model files
- Ensure `best_car_model.keras` and `class_mapping.json` exist

**Docker Issues:**
```bash
docker system prune -a  # Clear Docker cache
docker build --no-cache -t car-classification-service .  # Force rebuild
```

**GPU Setup (Optional):**
```bash
# Check GPU availability
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Use CPU-only if needed
export CUDA_VISIBLE_DEVICES=""
```

## 📝 Checklist Compliance

This project meets all requirements:

- ✅ **model_training.ipynb**: Complete training notebook with Python 3 kernel
- ✅ **Data Loading**: Stanford Cars dataset with preprocessing
- ✅ **Model Definition**: ResNet50 with TensorFlow-Keras framework
- ✅ **Training Loop**: Complete training with metrics visualization
- ✅ **Model Saving**: Exports to `.keras` and `.h5` format 
- ✅ **Environment Export**: `pip freeze` cell for reproducibility
- ✅ **API Service**: FastAPI with `/predict` endpoint
- ✅ **Docker Support**: Complete containerization
- ✅ **Documentation**: Comprehensive setup and usage instructions
- ✅ **GitHub Ready**: Clean repository structure for review
- ✅ **Automation**: `run.py` script for easy setup and deployment

## 📊 Dataset Information

The Stanford Cars Dataset contains:
- **16,185 images** of cars
- **196 classes** (make, model, year combinations)
- **Training set**: 8,144 images
- **Test set**: 8,041 images

Download from [Kaggle](https://www.kaggle.com/datasets/cyizhuo/stanford-cars-by-classes-folder) (dataset not included due to licensing).

## 🔒 GitHub Access

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stanford Cars Dataset creators
- TensorFlow and Keras teams
- FastAPI framework
- ResNet architecture authors

## 📞 Contact

For questions or support, please open an issue in the GitHub repository.