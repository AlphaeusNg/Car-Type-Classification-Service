"""
Car Type Classification API
FastAPI service for predicting car make/model from images
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import sys
import time
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api.request_metrics import RequestMetrics
    from api.utils import (
        decode_predictions,
        load_class_mapping,
        load_model,
        preprocess_image,
        validate_runtime_artifacts,
    )
except ImportError:
    # Fallback for when running from api directory
    from request_metrics import RequestMetrics
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
DETAIL_UNSUPPORTED_MEDIA = "File must be a JPEG or PNG image"
DETAIL_EMPTY_IMAGE = "Image file is empty"
DETAIL_OVERSIZED_UPLOAD = "Image exceeds the 10 MB upload limit"
DETAIL_OVERSIZED_REQUEST = "Request exceeds the upload size limit"
DETAIL_INVALID_IMAGE = "Invalid image data"
DETAIL_MODEL_NOT_READY = "Model is not ready"
DETAIL_IMAGE_BUSY = "Image processing queue is busy; retry later"
DETAIL_PREDICTION_BUSY = "Prediction queue is busy; retry later"
image_processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_PROCESSING)
prediction_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREDICTIONS)
request_metrics = RequestMetrics()
# In-flight lane workers. Shutdown waits for these and does not cancel them.
lane_tasks: set[asyncio.Task] = set()
shutting_down = asyncio.Event()


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
        request_metrics.record_rejection("oversized_request")
        response = JSONResponse(
            status_code=413,
            content={"detail": DETAIL_OVERSIZED_REQUEST},
        )
        await response(scope, receive, send)


def run_model_inference(loaded_model, processed_image, index_to_class):
    """Run and decode one synchronous model prediction."""
    predictions = loaded_model.predict(processed_image, verbose=0)
    return decode_predictions(predictions, index_to_class)


def _retrieve_task_outcome(task: asyncio.Task) -> None:
    """Mark a detached lane task retrieved so a cancelled client logs nothing."""
    lane_tasks.discard(task)
    if task.cancelled():
        return
    task.exception()


async def run_holding_lane(semaphore, timeout, operation, on_wait=None):
    """Run `operation` only after `semaphore` is acquired.

    A caller cancelled while still waiting does not keep a permit. A caller
    cancelled after acquire does not release the permit: the shielded worker
    finishes and releases it. That keeps exclusive access to the shared model
    when a client disconnects or the server shuts down.
    """
    started = time.perf_counter()
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
    except (asyncio.CancelledError, TimeoutError):
        if on_wait is not None:
            on_wait(time.perf_counter() - started)
        raise

    async def guarded():
        try:
            return await operation()
        finally:
            semaphore.release()

    try:
        if on_wait is not None:
            on_wait(time.perf_counter() - started)
        task = asyncio.create_task(guarded())
    except BaseException:
        semaphore.release()
        raise
    lane_tasks.add(task)
    task.add_done_callback(_retrieve_task_outcome)
    return await asyncio.shield(task)


@asynccontextmanager
async def lifespan(_app):
    """Load inference dependencies before serving requests."""
    global model, class_mapping, image_processing_semaphore, prediction_semaphore
    global shutting_down
    model = None
    class_mapping = None
    shutting_down = asyncio.Event()

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
    try:
        yield
    finally:
        # Wait for owned lane work. Do not cancel it and do not drop its permit.
        shutting_down.set()
        pending = [task for task in list(lane_tasks) if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


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


@app.get("/metrics")
async def metrics():
    """Operator aggregates. Counts and durations only; no request contents."""
    return request_metrics.snapshot()

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
        request_metrics.record_unavailable("model_not_ready")
        raise HTTPException(status_code=503, detail=DETAIL_MODEL_NOT_READY)

    declared_image_type = (image.content_type or "").partition(";")[0].strip().lower()
    if declared_image_type not in ALLOWED_IMAGE_TYPES:
        request_metrics.record_rejection("unsupported_media_type")
        raise HTTPException(status_code=400, detail=DETAIL_UNSUPPORTED_MEDIA)

    image_data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_data:
        request_metrics.record_rejection("empty_image")
        raise HTTPException(status_code=400, detail=DETAIL_EMPTY_IMAGE)
    if len(image_data) > MAX_UPLOAD_BYTES:
        request_metrics.record_rejection("oversized_upload")
        raise HTTPException(status_code=413, detail=DETAIL_OVERSIZED_UPLOAD)

    async def preprocess_for_request():
        started = time.perf_counter()
        try:
            return await run_in_threadpool(preprocess_image, image_data)
        finally:
            request_metrics.record_timing(
                "preprocessing", time.perf_counter() - started
            )

    try:
        processed_image = await run_holding_lane(
            image_processing_semaphore,
            IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS,
            preprocess_for_request,
        )
    except TimeoutError as exc:
        request_metrics.record_unavailable("image_processing_busy")
        logger.warning(
            "Image processing queue wait exceeded %.1f seconds",
            IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=503,
            detail=DETAIL_IMAGE_BUSY,
            headers={"Retry-After": str(IMAGE_PROCESSING_RETRY_AFTER_SECONDS)},
        ) from exc
    except ValueError as exc:
        request_metrics.record_rejection("invalid_image")
        logger.warning("Rejected invalid image upload")
        raise HTTPException(status_code=400, detail=DETAIL_INVALID_IMAGE) from exc

    async def infer():
        started = time.perf_counter()
        try:
            return await run_in_threadpool(
                run_model_inference,
                model,
                processed_image,
                class_mapping["index_to_class"],
            )
        finally:
            request_metrics.record_timing("inference", time.perf_counter() - started)

    def record_inference_wait(seconds: float) -> None:
        request_metrics.record_timing("inference_queue_wait", seconds)

    try:
        # Do not time out this worker: TensorFlow threads cannot be abandoned safely.
        decoded = await run_holding_lane(
            prediction_semaphore,
            PREDICTION_QUEUE_TIMEOUT_SECONDS,
            infer,
            on_wait=record_inference_wait,
        )
        return {**decoded, "status": "success"}

    except TimeoutError as exc:
        request_metrics.record_unavailable("prediction_queue_busy")
        logger.warning(
            "Prediction queue wait exceeded %.1f seconds",
            PREDICTION_QUEUE_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=503,
            detail=DETAIL_PREDICTION_BUSY,
            headers={"Retry-After": str(PREDICTION_RETRY_AFTER_SECONDS)},
        ) from exc
    except Exception as exc:
        logger.error("Prediction failed: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Global exception type=%s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status": "error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
