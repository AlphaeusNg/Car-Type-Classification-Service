"""
Car Type Classification API
FastAPI service for predicting car make/model from images
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.utils import (
        PREPROCESSED_IMAGE_SHAPE,
        decode_predictions,
        load_class_mapping,
        load_model,
        preprocess_image,
        validate_class_mapping,
    )
except ImportError:
    # Fallback for when running from api directory
    from utils import (
        PREPROCESSED_IMAGE_SHAPE,
        decode_predictions,
        load_class_mapping,
        load_model,
        preprocess_image,
        validate_class_mapping,
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and class mapping
model = None
class_mapping = None
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_CONCURRENT_IMAGE_PROCESSING = 2
IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS = 1.0
IMAGE_PROCESSING_RETRY_AFTER_SECONDS = 1
MAX_CONCURRENT_PREDICTIONS = 1
PREDICTION_QUEUE_TIMEOUT_SECONDS = 5.0
PREDICTION_RETRY_AFTER_SECONDS = 5
image_processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_PROCESSING)
prediction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)


def run_model_inference(loaded_model, processed_image, index_to_class):
    """Run and decode one synchronous model prediction."""
    predictions = loaded_model.predict(processed_image, verbose=0)
    return decode_predictions(predictions, index_to_class)


def validate_runtime_artifacts(loaded_model, loaded_mapping):
    """Reject incompatible model and class-mapping artifacts at startup."""
    validate_class_mapping(loaded_mapping)
    index_to_class = loaded_mapping["index_to_class"]

    input_shape = getattr(loaded_model, "input_shape", None)
    if not isinstance(input_shape, (tuple, list)) or not input_shape:
        raise ValueError("model must expose a single input shape")
    if isinstance(input_shape[0], (tuple, list)):
        raise ValueError("multi-input models are not supported")
    if len(input_shape) != len(PREPROCESSED_IMAGE_SHAPE):
        raise ValueError("model input shape must have rank 4")
    for actual, expected in zip(input_shape, PREPROCESSED_IMAGE_SHAPE):
        if actual is None:
            continue
        try:
            actual = int(actual)
        except (TypeError, ValueError) as exc:
            raise ValueError("model input dimensions must be known or dynamic") from exc
        if actual != expected:
            raise ValueError(
                f"model input shape {input_shape} does not accept "
                f"preprocessed shape {PREPROCESSED_IMAGE_SHAPE}"
            )

    output_shape = getattr(loaded_model, "output_shape", None)
    if not isinstance(output_shape, (tuple, list)) or not output_shape:
        raise ValueError("model must expose a single output shape")
    if isinstance(output_shape[0], (tuple, list)):
        raise ValueError("multi-output models are not supported")
    if len(output_shape) != 2:
        raise ValueError("model output shape must have rank 2")
    output_batch = output_shape[0]
    if output_batch is not None:
        try:
            output_batch = int(output_batch)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model output batch dimension must be known or dynamic"
            ) from exc
        if output_batch != 1:
            raise ValueError("model output must describe one score row per image")
    try:
        output_width = int(output_shape[-1])
    except (TypeError, ValueError) as exc:
        raise ValueError("model output width must be known") from exc
    if output_width <= 0:
        raise ValueError("model output width must be positive")

    if len(index_to_class) != output_width:
        raise ValueError("model output width does not match class mapping")


@asynccontextmanager
async def lifespan(_app):
    """Load inference dependencies before serving requests."""
    global model, class_mapping, image_processing_semaphore, prediction_semaphore
    model = None
    class_mapping = None

    try:
        logger.info("🚀 Loading model and class mapping...")
        loaded_model = load_model()
        loaded_mapping = load_class_mapping()
        validate_runtime_artifacts(loaded_model, loaded_mapping)
        model = loaded_model
        class_mapping = loaded_mapping
        image_processing_semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_IMAGE_PROCESSING
        )
        prediction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
        logger.info("✅ Model and class mapping loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
    yield


# Initialize FastAPI
app = FastAPI(
    title="🚗 Car Type Classification API",
    description="AI service to identify car make/model/year from photos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


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
        await asyncio.wait_for(
            image_processing_semaphore.acquire(),
            timeout=IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        logger.warning(
            "Image processing queue wait exceeded %.1f seconds",
            IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=503,
            detail="Image processing queue is busy; retry later",
            headers={"Retry-After": str(IMAGE_PROCESSING_RETRY_AFTER_SECONDS)},
        ) from exc

    try:
        # Do not time out this worker: threadpool work cannot be abandoned safely.
        processed_image = await run_in_threadpool(preprocess_image, image_data)
    except ValueError as exc:
        logger.warning("Rejected invalid image upload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid image data") from exc
    finally:
        image_processing_semaphore.release()

    try:
        await asyncio.wait_for(
            prediction_semaphore.acquire(),
            timeout=PREDICTION_QUEUE_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        logger.warning(
            "Prediction queue wait exceeded %.1f seconds",
            PREDICTION_QUEUE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=503,
            detail="Prediction queue is busy; retry later",
            headers={"Retry-After": str(PREDICTION_RETRY_AFTER_SECONDS)},
        ) from exc

    try:
        # Do not time out this worker: TensorFlow threads cannot be abandoned safely.
        decoded = await run_in_threadpool(
            run_model_inference,
            model,
            processed_image,
            class_mapping["index_to_class"],
        )
        return {**decoded, "status": "success"}

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    finally:
        prediction_semaphore.release()


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
