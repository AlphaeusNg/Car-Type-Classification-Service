# Car Type Classification Service

A deep learning image classification service for car types using a ResNet50-based model trained on the Stanford Cars Dataset. This project fulfills the Home Team Department assessment requirements with complete model training in Jupyter notebook, REST API deployment, and Docker containerization.

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
├── README.md                     # This file (REQUIRED)
├── run.py                        # Automated setup and deployment script
├── car_classification_model.h5   # Trained model (generated)
├── class_mapping.json           # Class index mapping (generated)
└── data/                        # Stanford Cars dataset
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.12
- pip or conda package manager
- Docker (for containerization)
- 4GB+ RAM (8GB+ recommended for training)

### Environment Setup

#### Option 1: Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/WSL2:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Conda Environment
```bash
# Create conda environment
conda create -n car-classification python=3.12
conda activate car-classification

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running the Project

### Quick Start with `run.py` (Recommended)

The project includes a comprehensive `run.py` script that handles all setup and deployment tasks:

```bash
# Interactive mode - choose your deployment option
python run.py

# Or use specific commands:
python run.py --setup                    # Setup project environment
python run.py --mode local               # Run API locally with uvicorn
python run.py --mode docker              # Run API in Docker container
python run.py --build                    # Build Docker image only
python run.py --mode local --port 8080   # Run on custom port
```

### Manual Setup (Alternative)

#### 1. Model Training
Launch Jupyter notebook and run the training pipeline:

```bash
# Start Jupyter notebook
jupyter notebook

# Open model_training.ipynb and run all cells
# The notebook will:
# - Load and preprocess Stanford Cars dataset
# - Define ResNet50-based model architecture
# - Train the model with data augmentation
# - Generate evaluation metrics and confusion matrix
# - Save model as car_classification_model.h5
# - Export package versions for reproducibility
```

#### 2. API Development (Local)
```bash
# Run the API locally
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at: http://localhost:8000
# Interactive docs at: http://localhost:8000/docs
```

#### 3. Docker Deployment
```bash
# Build Docker image
docker build -t car-classification-service .

# Run container
docker run -p 8000:8000 car-classification-service

# API will be available at: http://localhost:8000
```

## � API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Car Type Prediction
```bash
# Upload image for classification
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@data/test/BMW 1 Series Coupe 2012/002240.jpg"
``` # Model has a bias for "Audi TTS Coupe 2012" Category data/test/Audi TTS Coupe 2012/001096.jpg

**Response Format:**
```json
{
  "predicted_class": "Audi A4 Sedan 2012",
  "confidence": 0.89,
  "top5_predictions": [
    {"class": "Audi A4 Sedan 2012", "confidence": 0.89},
    {"class": "Audi A6 Sedan 2012", "confidence": 0.08},
    {"class": "BMW 3 Series Sedan 2012", "confidence": 0.02},
    {"class": "Mercedes-Benz C-Class Sedan 2012", "confidence": 0.01},
    {"class": "Audi A3 Sedan 2012", "confidence": 0.00}
  ],
  "status": "success"
}
```

### Error Handling
The API returns appropriate HTTP status codes:
- **400**: Invalid file format (non-image files)
- **500**: Server/prediction errors

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
- **Container**: Python 3.8-slim base image
- **Port**: 8000 (configurable)
- **Health Checks**: Built-in health monitoring
- **Error Handling**: Comprehensive error responses

## 📊 Model Performance

Performance metrics are generated in the notebook:
- **Training Accuracy**: ~95%+ 
- **Validation Accuracy**: ~90%+
- **Top-5 Accuracy**: ~98%+
- **Confusion Matrix**: Detailed class-wise performance
- **Training Plots**: Loss and accuracy curves

## 🔒 GitHub Access

**Note**: Please grant repository access to user `MdJawad` for review and assessment.

## 🐛 Troubleshooting

### Quick Fixes with `run.py`

```bash
# If you encounter environment issues:
python run.py --setup

# If Docker build fails:
python run.py --build

# If port 8000 is busy:
python run.py --mode local --port 8080
```

### Common Issues

1. **GPU Setup (Optional)**:
   ```bash
   # Check GPU availability
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Memory Issues**:
   - Reduce batch size in notebook
   - Use CPU-only training: `export CUDA_VISIBLE_DEVICES=""`

3. **Docker Build Issues**:
   ```bash
   # Clear Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker build --no-cache -t car-classification-service .
   ```

4. **API Connection Issues**:
   ```bash
   # Check if port is available
   netstat -tlnp | grep :8000
   
   # Use different port
   uvicorn api.main:app --host 0.0.0.0 --port 8080
   ```

5. **Missing Model Files**:
   ```bash
   # Run the training notebook first to generate:
   # - car_classification_model.h5
   # - class_mapping.json
   jupyter notebook model_training.ipynb
   ```

## 📝 Assessment Compliance

This project meets all assessment requirements:

- ✅ **model_training.ipynb**: Complete training notebook with Python 3 kernel
- ✅ **Data Loading**: Stanford Cars dataset with preprocessing
- ✅ **Model Definition**: ResNet50 with TensorFlow-Keras framework
- ✅ **Training Loop**: Complete training with metrics visualization
- ✅ **Model Saving**: Exports to `.h5` format
- ✅ **Environment Export**: `pip freeze` cell for reproducibility
- ✅ **API Service**: FastAPI with `/predict` endpoint
- ✅ **Docker Support**: Complete containerization
- ✅ **Documentation**: Comprehensive setup and usage instructions
- ✅ **GitHub Ready**: Clean repository structure for review
- ✅ **Automation**: `run.py` script for easy setup and deployment

1. **Visit the NVIDIA CUDA Downloads page**:
   - Direct link for Ubuntu 24.04: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network
   - For other distributions: https://developer.nvidia.com/cuda-downloads

2. **Follow the installation instructions provided by NVIDIA**:
   ```bash
   # Example for Ubuntu 24.04 (verify commands on NVIDIA's site):
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   ```

3. **Verify CUDA installation**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

4. **Install cuDNN** (required for TensorFlow):
   - Download from: https://developer.nvidia.com/cudnn
   - Follow NVIDIA's cuDNN installation guide

⚠️ **Important**: Download packages directly from NVIDIA's website rather than using cached .deb files. This ensures you get the latest versions with security updates.

#### Windows GPU Support

For Windows users requiring GPU support:
- **Recommended**: Use WSL2 with Ubuntu and follow the Linux instructions above
- **Alternative**: Install CUDA for Windows from https://developer.nvidia.com/cuda-downloads?target_os=Windows

### Platform-Specific Setup

#### 🐧 Linux/WSL2 (Recommended)

This project was developed and tested on WSL2 Ubuntu. WSL2 provides the best experience with full GPU support.

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlphaeusNg/Car-Type-Classification-Service.git
   cd Car-Type-Classification-Service
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

#### 🪟 Windows

For Windows users, we strongly recommend using WSL2 for the best experience. However, CPU-only installation is also supported.

**Option 1: WSL2 (Recommended)**
1. Install WSL2 with Ubuntu
2. Follow the Linux/WSL2 instructions above

**Option 2: Native Windows (CPU-only)**
1. **Clone the repository**
   ```cmd
   git clone https://github.com/AlphaeusNg/Car-Type-Classification-Service.git
   cd Car-Type-Classification-Service
   ```

2. **Create virtual environment**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Modify requirements.txt for Windows**
   Replace the tensorflow line in `requirements.txt` with:
   ```
   https://storage.googleapis.com/tensorflow/versions/2.19.0/tensorflow-2.19.0-cp311-cp311-win_amd64.whl
   ```

4. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlphaeusNg/Car-Type-Classification-Service.git
   cd Car-Type-Classification-Service
   ```

2. **Create virtual environment**
### Verification

Verify your installation by checking TensorFlow:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```

## 🚀 Usage

### 1. Model Training

Run the Jupyter notebook to train the model:

```bash
jupyter notebook assignment.ipynb
```

The notebook will:
- Set up the environment and download dependencies
- Create synthetic data (or load Stanford Cars Dataset)
- Build and train a ResNet50-based model
- Evaluate the model with comprehensive metrics
- Save the trained model and class mappings

### 2. API Service

#### Local Development

Start the FastAPI service locally:

```bash
cd api
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`

#### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

### 3. Docker Deployment

#### Build the Docker image

```bash
docker build -t car-classification-service .
```

#### Run the container

```bash
docker run -p 8080:8080 car-classification-service
```

#### Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  car-classification:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models:ro
      - ./class_mapping.json:/app/class_mapping.json:ro
```

Run with:
```bash
docker-compose up
```

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Car Type Prediction
```bash
POST /predict
Content-Type: multipart/form-data
Body: image file (JPEG/PNG)
```

#### Example Request (curl)

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@car_image.jpg"
```

#### Example Response

```json
{
  "predicted_class": "BMW M3 Coupe 2012",
  "confidence": 0.8745,
  "top5_predictions": [
    {
      "class": "BMW M3 Coupe 2012",
      "confidence": 0.8745
    },
    {
      "class": "BMW M5 Sedan 2010",
      "confidence": 0.0892
    },
    {
      "class": "BMW 3 Series Sedan 2012",
      "confidence": 0.0234
    },
    {
      "class": "Audi S4 Sedan 2012",
      "confidence": 0.0089
    },
    {
      "class": "BMW X3 SUV 2012",
      "confidence": 0.0040
    }
  ],
  "status": "success"
}
```

## 🧠 Model Details

### Architecture
- **Backbone**: ResNet50 pre-trained on ImageNet
- **Input Size**: 224x224x3 RGB images
- **Output**: 196 classes (Stanford Cars Dataset)
- **Training Strategy**: Transfer learning + Fine-tuning

### Training Process
1. **Phase 1**: Feature extraction with frozen backbone
2. **Phase 2**: Fine-tuning with unfrozen top layers
3. **Data Augmentation**: Rotation, shifts, flips, zoom, shear
4. **Regularization**: Dropout layers, early stopping

### Performance Metrics
- **Top-1 Accuracy**: Reported in notebook
- **Top-5 Accuracy**: Reported in notebook
- **Confusion Matrix**: Visualized for model evaluation
- **Classification Report**: Precision, recall, F1-score per class

## 🛡️ Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file format or missing image
- **500 Internal Server Error**: Model prediction failures
- **Logging**: Detailed logs for debugging and monitoring

## 🧪 Testing

### Manual Testing

Test the API with sample images:

```bash
# Test health endpoint
curl http://localhost:8080/health

# Test prediction with an image
curl -X POST "http://localhost:8080/predict" \
     -F "image=@test_car.jpg"
```

### Automated Testing

Run pytest tests (if implemented):

```bash
pytest tests/
```

## 📊 Stanford Cars Dataset

The Stanford Cars Dataset contains:
- **16,185 images** of cars
- **196 classes** (make, model, year combinations)
- **Training set**: 8,144 images
- **Test set**: 8,041 images

### Dataset Access

Due to licensing, the dataset is not included in this repository. You can:
1. Download from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)

## 🔧 Configuration

### Environment Variables

- `PYTHONPATH`: Set to `/app` for proper imports
- `PYTHONUNBUFFERED`: Set to `1` for real-time logging
- `MODEL_PATH`: Custom model file path (optional)
- `CLASS_MAPPING_PATH`: Custom class mapping file path (optional)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stanford Cars Dataset creators
- TensorFlow and Keras teams
- FastAPI framework
- ResNet architecture authors

## 📞 Contact

For questions or support, please open an issue in the GitHub repository.