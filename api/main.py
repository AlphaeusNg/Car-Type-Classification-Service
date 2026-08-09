"""
Car Type Classification API
FastAPI service for predicting car make/model from images
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import logging
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.utils import preprocess_image, load_model, load_class_mapping
except ImportError:
    # Fallback for when running from api directory
    from utils import preprocess_image, load_model, load_class_mapping

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="🚗 Car Type Classification API",
    description="AI service to identify car make/model/year from photos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model and class mapping
model = None
class_mapping = None
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

@app.on_event("startup")
async def startup_event():
    """Load model and class mapping on startup"""
    global model, class_mapping
    
    try:
        logger.info("🚀 Loading model and class mapping...")
        model = load_model()
        class_mapping = load_class_mapping()
        logger.info("✅ Model and class mapping loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Car Type Classification Service is running!"}

@app.get("/health")
async def health_check():
    """Readiness check for the model and class mapping."""
    ready = model is not None and class_mapping is not None
    payload = {
        "status": "healthy" if ready else "unavailable",
        "model_loaded": model is not None,
        "class_mapping_loaded": class_mapping is not None,
        "total_classes": len(class_mapping.get("index_to_class", {})) if class_mapping else 0
    }
    if not ready:
        return JSONResponse(status_code=503, content=payload)
    return payload

@app.post("/predict")
async def predict_car_type(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict car type from uploaded image
    
    Args:
        image: Uploaded image file (JPEG/PNG)
        
    Returns:
        JSON with predicted class, confidence, and top-5 predictions
    """
    if model is None or class_mapping is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="File must be a JPEG or PNG image")

    image_data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_data:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")

    try:
        processed_image = preprocess_image(image_data)
    except ValueError as exc:
        logger.warning("Rejected invalid image upload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    try:
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top prediction
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = class_mapping['index_to_class'][str(predicted_idx)]
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_predictions = [
            {
                "class": class_mapping['index_to_class'][str(idx)],
                "confidence": float(predictions[0][idx])
            }
            for idx in top5_indices
        ]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top5_predictions": top5_predictions,
            "status": "success"
        }
        
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
