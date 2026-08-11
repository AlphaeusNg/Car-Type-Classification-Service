# Car Type Classification Service

A production-ready deep learning service that classifies car images into 196 car types using ResNet50 transfer learning.

## 🎯 Overview

This service classifies car images into 196 different car types using transfer learning with ResNet50 backbone:

- **🧠 Model**: ResNet50 + transfer learning (TensorFlow 2.19.0)
- **📊 Dataset**: Stanford Cars (196 classes, 16K+ images)
- **🚀 API**: FastAPI with `/predict` endpoint
- **🐳 Deploy**: Docker containerization
- **📝 Training**: Complete Jupyter notebook pipeline

## 🏗️ Project Structure

```
├── model_training.ipynb          # 📓 Complete training pipeline
├── gpu_setup_guide.ipynb         # 📓 GPU setup and debugging
├── api/                          # 🚀 FastAPI service
│   ├── main.py                   #   └── REST API endpoints
│   └── utils.py                  #   └── Model utilities
├── run.py                        # ⚡ One-command setup & deploy
├── requirements.txt              # 📦 Python dependencies
├── README.md                     # 📄Documentation
├── Dockerfile                    # 🐳 Container configuration
├── best_car_model.keras          # 🧠 Trained model (auto-generated)
├── class_mapping.json            # 🏷️ Class labels (auto-generated)
└── data/                         # 📁 Stanford Cars dataset
    ├── train/                    #   └── Training images (8K)
    └── test/                     #   └── Test images (8K)
```

## ⚡ Quick Start

### Prerequisites
- Python 3.12+ (recommended: 3.12)
- 4GB RAM (8GB+ recommended for training)
- Docker (optional)

### 1. One-Command Setup
```bash
git clone https://github.com/AlphaeusNg/Car-Type-Classification-Service.git
cd Car-Type-Classification-Service
python3 run.py --setup  # Auto-installs everything
```

### 2. Train Model
```bash
# Jupyter Notebook (Interactive)
jupyter notebook model_training.ipynb

```

### 3. Start API
```bash
python3 run.py --mode local    # Local development
python3 run.py --mode docker   # Docker deployment
```

That's it! API runs at **http://localhost:8000** 🎉

### Run tests

```bash
python3 -m pytest -q
```

The tests use a lightweight fake predictor, so trained model files are not
required to verify health, upload validation, error handling, and response
ranking. CI installs `requirements-test.txt`; TensorFlow remains a runtime and
training dependency and is imported only when a real model is loaded.

Dependency manifests are intentionally scoped:

- `requirements-api.txt`: production API/Docker inference only.
- `requirements-test.txt`: fast model-free contract tests and CI.
- `requirements.txt`: complete training, notebook, analysis, and API workspace.

`run.py` and the Docker build accept the same model formats as the API loader:
`best_car_model.keras`, `car_classification_model.h5`, or
`models/car_classification_savedmodel`. For a manual Docker build, select the
artifact explicitly when it is not the default:

```bash
docker build --build-arg MODEL_PATH=car_classification_model.h5 -t car-type-clf .
```

## 🔌 API Usage

### Test the Service
```bash
# Health check
curl http://localhost:8000/health

# Predict car type
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@data/test/Acura TL Sedan 2012/000197.jpg"
```

`/health` is a readiness check: it returns HTTP 503 until both the model and
class mapping are loaded, input/output shapes match preprocessing and class
labels, and then returns HTTP 200 with `"status": "healthy"`.

### Response Format
```json
{
  "predicted_class": "Acura TL Sedan 2012",
  "confidence": 0.2989,
  "top5_predictions": [
    {"class": "Acura TL Sedan 2012", "confidence": 0.2989},
    {"class": "Chevrolet Malibu Hybrid Sedan 2010", "confidence": 0.2387},
    {"class": "Cadillac SRX SUV 2012", "confidence": 0.0466}
  ],
  "status": "success"
}
```

At startup, model output metadata must describe a rank-two tensor with a dynamic
or single-image batch. The API accepts inference output only when it is one
non-empty, finite numeric score row whose width matches `class_mapping.json`.
Invalid model output is logged server-side and returned as the generic
`Prediction failed` response.
The decoded file format must also be JPEG or PNG; renaming another image type or
changing only its upload MIME type is rejected. JPEG orientation metadata is
applied before resize, so phone photos reach the model in their displayed
orientation.

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧠 Model Details

### Architecture
- **Base**: ResNet50 (pre-trained on ImageNet)
- **Input**: 224×224 RGB images  
- **Output**: 196 car classes (softmax)
- **Size**: ~172MB

### Training Pipeline
1. **Phase 1** (~25 epochs): Frozen backbone + train classifier
2. **Phase 2** (~15 epochs): Fine-tune top layers with lower LR

### Performance Metrics
| Metric | Score |
|--------|-------|
| Training Accuracy | 99%+ |
| Validation Accuracy | 58.69% |
| Top-5 Accuracy | 84.39% |
| Model Size | 172.64MB |

## � Troubleshooting

### Common Issues

**🚨 "Model not found"**
```bash
# Train model first
jupyter notebook model_training.ipynb
# Or use: python refactored_training.py
```

**🚨 "Port already in use"**
```bash
python run.py --mode local --port 8080  # Use different port
```

**🚨 "Environment issues"**
```bash
python run.py --setup  # Reset environment
```

**🚨 "Docker problems"**
```bash
docker system prune -a  # Clear cache
docker build --no-cache -t car-classification-service .
```

### GPU Setup (Optional)
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

## ✅ Features Checklist

**Core Requirements**
- ✅ `model_training.ipynb` - Interactive training notebook
- ✅ `refactored_training.py` - Production training script  
- ✅ Stanford Cars dataset support (196 classes)
- ✅ ResNet50 + transfer learning architecture
- ✅ Complete training pipeline with metrics
- ✅ Model export (`.keras` + `.h5` formats)
- ✅ FastAPI service with `/predict` endpoint
- ✅ Docker containerization support
- ✅ Environment reproducibility (`requirements.txt`)
- ✅ Comprehensive documentation
- ✅ One-command setup script (`run.py`)

**Bonus Features**
- ✅ Two-phase training (base + fine-tuning)
- ✅ Mixed precision training support
- ✅ TensorBoard integration  
- ✅ Health check endpoints
- ✅ Top-5 predictions
- ✅ Swagger/OpenAPI documentation
- ✅ GPU/CPU auto-detection

## 📊 Dataset

**Stanford Cars Dataset**
- 📸 **16,185 images** total
- 🏷️ **196 classes** (make/model/year)  
- 🎯 **8,144 training** + **8,041 test** images
- 📥 Download from [Kaggle](https://www.kaggle.com/datasets/cyizhuo/stanford-cars-by-classes-folder)

*Note: Dataset not included due to size/licensing*

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Credits

- Stanford Cars Dataset creators
- TensorFlow/Keras teams  
- FastAPI framework
- ResNet architecture authors

## 📞 Contact

For questions or support, please open an issue in the GitHub repository.
