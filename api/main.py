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
        decode_predictions,
        load_class_mapping,
        load_model,
        preprocess_image,
        validate_runtime_artifacts,
    )
except ImportError:
    # Fallback for when running from api directory
    from utils import (
        decode_predictions,
        load_class_mapping,
        load_model,
        preprocess_image,
        validate_runtime_artifacts,
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and class mapping
model = None
class_mapping = None
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
MAX_CONCURRENT_IMAGE_PROCESSING = 2
IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS = 1.0
IMAGE_PROCESSING_RETRY_AFTER_SECONDS = 1
MAX_CONCURRENT_PREDICTIONS = 1
PREDICTION_QUEUE_TIMEOUT_SECONDS = 5.0
PREDICTION_RETRY_AFTER_SECONDS = 5
image_processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_PROCESSING)
prediction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)


class RequestBodyTooLarge(Exception):
    """Internal signal raised before multipart parsing exceeds its budget."""


class PredictRequestBodyLimitMiddleware:
    """Bound `/predict` request bytes before Starlette spools uploaded files."""

    def __init__(self, wrapped_app):
        self.wrapped_app = wrapped_app

    async def __call__(self, scope, receive, send):
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in {"/predict", "/predict/"}
        ):
            await self.wrapped_app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                declared_length = int(value)
            except (TypeError, ValueError):
                break
            if declared_length > MAX_REQUEST_BODY_BYTES:
                await self._reject(scope, receive, send)
                return

        received_bytes = 0

        async def receive_limited():
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_REQUEST_BODY_BYTES:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.wrapped_app(scope, receive_limited, send)
        except RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send):
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request exceeds the upload size limit"},
        )
        await response(scope, receive, send)


def run_model_inference(loaded_model, processed_image, index_to_class):
    """Run and decode one synchronous model prediction."""
    predictions = loaded_model.predict(processed_image, verbose=0)
    return decode_predictions(predictions, index_to_class)


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
app.add_middleware(PredictRequestBodyLimitMiddleware)


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
