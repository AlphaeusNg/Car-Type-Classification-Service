import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from io import BytesIO
from pathlib import Path
from threading import Event, Lock

import httpx2 as httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api
from api.utils import preprocess_image


class FakeModel:
    def __init__(self, predictions=None, error=None):
        self.predictions = predictions
        self.error = error
        self.input_shape = (None, 224, 224, 3)
        self.output_shape = (None, len(predictions)) if predictions is not None else (None, 5)

    def predict(self, _image, verbose=0):
        assert verbose == 0
        if self.error:
            raise self.error
        return np.array([self.predictions], dtype=np.float32)


class RawOutputModel:
    output_shape = (None, 5)

    def __init__(self, output):
        self.output = output

    def predict(self, _image, verbose=0):
        assert verbose == 0
        return self.output


class BlockingModel(FakeModel):
    def __init__(self, predictions):
        super().__init__(predictions)
        self.first_prediction_started = Event()
        self.release_predictions = Event()
        self.wait_timeout = 5
        self._state_lock = Lock()
        self.active_predictions = 0
        self.max_active_predictions = 0

    def predict(self, image, verbose=0):
        with self._state_lock:
            self.active_predictions += 1
            self.max_active_predictions = max(
                self.max_active_predictions, self.active_predictions
            )
            self.first_prediction_started.set()

        try:
            if not self.release_predictions.wait(timeout=self.wait_timeout):
                raise TimeoutError("test prediction was not released")
            return super().predict(image, verbose=verbose)
        finally:
            with self._state_lock:
                self.active_predictions -= 1


class BlockingPreprocessor:
    def __init__(self, expected_concurrency):
        self.expected_concurrency = expected_concurrency
        self.expected_calls_started = Event()
        self.release_calls = Event()
        self._state_lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def __call__(self, _image_data):
        with self._state_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            if self.active_calls == self.expected_concurrency:
                self.expected_calls_started.set()

        try:
            if not self.release_calls.wait(timeout=5):
                raise TimeoutError("test preprocessing was not released")
            return np.zeros((1, 224, 224, 3), dtype=np.float32)
        finally:
            with self._state_lock:
                self.active_calls -= 1


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch):
    monkeypatch.setattr(api, "model", None)
    monkeypatch.setattr(api, "class_mapping", None)
    api.request_metrics.reset()
    api.lane_tasks.clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def make_ready(monkeypatch, predictions=None):
    values = predictions or [0.05, 0.1, 0.6, 0.15, 0.1]
    monkeypatch.setattr(api, "model", FakeModel(values))
    monkeypatch.setattr(
        api,
        "class_mapping",
        {"index_to_class": {str(index): f"class-{index}" for index in range(5)}},
    )
    monkeypatch.setattr(
        api,
        "preprocess_image",
        lambda _data: np.zeros((1, 224, 224, 3), dtype=np.float32),
    )


def mapping(size=5):
    labels = {str(index): f"class-{index}" for index in range(size)}
    return {
        "index_to_class": labels,
        "class_to_index": {label: int(index) for index, label in labels.items()},
    }


def test_health_is_unavailable_until_dependencies_load(client):
    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "model_loaded": False,
        "class_mapping_loaded": False,
        "total_classes": 0,
    }


def test_health_reports_ready_model(client, monkeypatch):
    make_ready(monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["total_classes"] == 5


def test_lifespan_loads_dependencies_before_serving(monkeypatch):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    loaded_mapping = mapping()
    monkeypatch.setattr(api, "load_model", lambda: loaded_model)
    monkeypatch.setattr(api, "load_class_mapping", lambda: loaded_mapping)

    with TestClient(api.app) as lifecycle_client:
        response = lifecycle_client.get("/health")

    assert response.status_code == 200
    assert api.model is loaded_model
    assert api.class_mapping is loaded_mapping


def test_runtime_artifacts_reject_model_mapping_width_mismatch():
    with pytest.raises(ValueError, match="output width does not match"):
        api.validate_runtime_artifacts(FakeModel([0.2, 0.3, 0.5]), mapping(5))


@pytest.mark.parametrize(
    ("output_shape", "message"),
    [
        ((5,), "rank 2"),
        ((None, 1, 5), "rank 2"),
        ((2, 5), "one score row"),
        (("many", 5), "batch dimension"),
    ],
)
def test_runtime_artifacts_reject_incompatible_model_output(output_shape, message):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    loaded_model.output_shape = output_shape

    with pytest.raises(ValueError, match=message):
        api.validate_runtime_artifacts(loaded_model, mapping())


@pytest.mark.parametrize("output_shape", [(None, 5), (1, 5)])
def test_runtime_artifacts_accept_compatible_model_output(output_shape):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    loaded_model.output_shape = output_shape

    api.validate_runtime_artifacts(loaded_model, mapping())


@pytest.mark.parametrize(
    ("input_shape", "message"),
    [
        (None, "single input shape"),
        ([(None, 224, 224, 3), (None, 1)], "multi-input"),
        ((None, 224, 224), "rank 4"),
        ((None, 299, 299, 3), "does not accept"),
        ((32, 224, 224, 3), "does not accept"),
    ],
)
def test_runtime_artifacts_reject_incompatible_model_input(input_shape, message):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    loaded_model.input_shape = input_shape

    with pytest.raises(ValueError, match=message):
        api.validate_runtime_artifacts(loaded_model, mapping())


@pytest.mark.parametrize(
    "input_shape",
    [(None, 224, 224, 3), (1, 224, 224, 3), (None, None, None, 3)],
)
def test_runtime_artifacts_accept_compatible_model_input(input_shape):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    loaded_model.input_shape = input_shape

    api.validate_runtime_artifacts(loaded_model, mapping())


def test_runtime_artifacts_reject_non_inverse_mapping():
    invalid_mapping = mapping()
    invalid_mapping["class_to_index"]["class-2"] = 4

    with pytest.raises(ValueError, match="must be the inverse"):
        api.validate_runtime_artifacts(FakeModel([0.05, 0.1, 0.6, 0.15, 0.1]), invalid_mapping)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["class_to_index"].update({"extra": 5}), "exactly"),
        (
            lambda value: (
                value["index_to_class"].__setitem__("0", 7),
                value["class_to_index"].pop("class-0"),
                value["class_to_index"].__setitem__(7, 0),
            ),
            "non-empty strings",
        ),
        (lambda value: value["class_to_index"].__setitem__("class-0", False), "integers"),
    ],
)
def test_runtime_artifacts_reject_non_bijective_mapping(mutate, message):
    invalid_mapping = mapping()
    mutate(invalid_mapping)

    with pytest.raises(ValueError, match=message):
        api.validate_runtime_artifacts(
            FakeModel([0.05, 0.1, 0.6, 0.15, 0.1]), invalid_mapping
        )


def test_lifespan_does_not_publish_incompatible_artifacts(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: FakeModel([0.2, 0.3, 0.5]))
    monkeypatch.setattr(api, "load_class_mapping", lambda: mapping(5))

    with pytest.raises(ValueError, match="output width does not match"):
        with TestClient(api.app):
            pass

    assert api.model is None
    assert api.class_mapping is None


def test_predict_returns_service_unavailable_before_model_load(client):
    response = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not ready"


def test_predict_rejects_oversized_request_before_form_parsing(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_REQUEST_BODY_BYTES", 4, raising=False)

    response = client.post(
        "/predict",
        content=b"12345",
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request exceeds the upload size limit"}


def test_predict_request_limit_counts_streamed_body_without_content_length(monkeypatch):
    monkeypatch.setattr(api, "MAX_REQUEST_BODY_BYTES", 4)
    request_messages = iter(
        [
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"45", "more_body": False},
        ]
    )
    downstream_messages = []
    response_messages = []

    async def receive():
        return next(request_messages)

    async def send(message):
        response_messages.append(message)

    async def downstream(_scope, limited_receive, _send):
        downstream_messages.append(await limited_receive())
        downstream_messages.append(await limited_receive())

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/predict",
        "raw_path": b"/predict",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    middleware = api.PredictRequestBodyLimitMiddleware(downstream)

    asyncio.run(middleware(scope, receive, send))

    assert len(downstream_messages) == 1
    assert response_messages[0]["status"] == 413
    assert b"Request exceeds the upload size limit" in response_messages[1]["body"]


def test_predict_rejects_unsupported_media_type(client, monkeypatch):
    make_ready(monkeypatch)

    response = client.post(
        "/predict",
        files={"image": ("car.gif", b"image", "image/gif")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be a JPEG or PNG image"


@pytest.mark.parametrize(
    "declared_content_type",
    ["Image/JPEG", "image/png; profile=srgb"],
)
def test_predict_accepts_case_insensitive_parameterized_image_media_type(
    client, monkeypatch, declared_content_type
):
    make_ready(monkeypatch)

    response = client.post(
        "/predict",
        files={"image": ("car", b"image", declared_content_type)},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("decoded_format", "claimed_content_type"),
    [("GIF", "image/jpeg"), ("WEBP", "image/png")],
)
def test_predict_rejects_unsupported_decoded_format(
    client, monkeypatch, decoded_format, claimed_content_type
):
    make_ready(monkeypatch)
    monkeypatch.setattr(api, "preprocess_image", preprocess_image)
    encoded = BytesIO()
    Image.new("RGB", (32, 16), "red").save(encoded, format=decoded_format)

    response = client.post(
        "/predict",
        files={"image": ("renamed-image", encoded.getvalue(), claimed_content_type)},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image data"


def test_predict_rejects_oversized_upload_without_preprocessing(client, monkeypatch):
    make_ready(monkeypatch)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    monkeypatch.setattr(api, "preprocess_image", lambda _data: pytest.fail("must not preprocess"))

    response = client.post(
        "/predict",
        files={"image": ("car.png", b"12345", "image/png")},
    )

    assert response.status_code == 413


def test_predict_rejects_invalid_image_without_exposing_decoder_error(client, monkeypatch):
    make_ready(monkeypatch)

    def reject_image(_data):
        raise ValueError("decoder internals")

    monkeypatch.setattr(api, "preprocess_image", reject_image)
    response = client.post(
        "/predict",
        files={"image": ("car.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image data"
    assert "decoder internals" not in response.text


def test_invalid_images_release_image_processing_capacity(monkeypatch):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    calls = 0

    def reject_twice_then_accept(_data):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise ValueError("invalid test image")
        return np.zeros((1, 224, 224, 3), dtype=np.float32)

    monkeypatch.setattr(api, "load_model", lambda: loaded_model)
    monkeypatch.setattr(api, "load_class_mapping", mapping)
    monkeypatch.setattr(api, "preprocess_image", reject_twice_then_accept)
    monkeypatch.setattr(api, "IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS", 0.05)

    with TestClient(api.app) as lifecycle_client:
        responses = [
            lifecycle_client.post(
                "/predict",
                files={"image": ("car.png", b"image", "image/png")},
            )
            for _ in range(3)
        ]

    assert [response.status_code for response in responses] == [400, 400, 200]


def test_predict_returns_ranked_classes(client, monkeypatch):
    make_ready(monkeypatch)

    response = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "class-2"
    assert body["confidence"] == pytest.approx(0.6)
    assert [item["class"] for item in body["top5_predictions"]] == [
        "class-2",
        "class-3",
        "class-4",
        "class-1",
        "class-0",
    ]


def test_predict_keeps_health_responsive_and_serializes_model_access(monkeypatch):
    loaded_model = BlockingModel([0.05, 0.1, 0.6, 0.15, 0.1])
    monkeypatch.setattr(api, "load_model", lambda: loaded_model)
    monkeypatch.setattr(api, "load_class_mapping", mapping)
    monkeypatch.setattr(
        api,
        "preprocess_image",
        lambda _data: np.zeros((1, 224, 224, 3), dtype=np.float32),
    )

    def request_prediction(lifecycle_client):
        return lifecycle_client.post(
            "/predict",
            files={"image": ("car.png", b"image", "image/png")},
        )

    health_while_blocked = None
    with TestClient(api.app) as lifecycle_client:
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(request_prediction, lifecycle_client)
            assert loaded_model.first_prediction_started.wait(timeout=1)
            second = executor.submit(request_prediction, lifecycle_client)
            health = executor.submit(lifecycle_client.get, "/health")

            try:
                health_while_blocked = health.result(timeout=1)
            except FutureTimeoutError:
                pass
            finally:
                loaded_model.release_predictions.set()

            prediction_responses = [
                first.result(timeout=2),
                second.result(timeout=2),
            ]
            health.result(timeout=2)

    assert health_while_blocked is not None
    assert health_while_blocked.status_code == 200
    assert all(response.status_code == 200 for response in prediction_responses)
    assert loaded_model.max_active_predictions == 1


def test_predict_bounds_queue_wait_without_abandoning_active_inference(monkeypatch):
    loaded_model = BlockingModel([0.05, 0.1, 0.6, 0.15, 0.1])
    monkeypatch.setattr(api, "load_model", lambda: loaded_model)
    monkeypatch.setattr(api, "load_class_mapping", mapping)
    monkeypatch.setattr(api, "PREDICTION_QUEUE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        api,
        "preprocess_image",
        lambda _data: np.zeros((1, 224, 224, 3), dtype=np.float32),
    )

    def request_prediction(lifecycle_client):
        return lifecycle_client.post(
            "/predict",
            files={"image": ("car.png", b"image", "image/png")},
        )

    with TestClient(api.app) as lifecycle_client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(request_prediction, lifecycle_client)
            assert loaded_model.first_prediction_started.wait(timeout=1)
            overloaded = executor.submit(request_prediction, lifecycle_client)

            try:
                overloaded_response = overloaded.result(timeout=1)
            finally:
                loaded_model.release_predictions.set()

            first_response = first.result(timeout=2)

        recovered_response = request_prediction(lifecycle_client)

    assert overloaded_response.status_code == 503
    assert overloaded_response.json() == {"detail": "Prediction queue is busy; retry later"}
    assert overloaded_response.headers["retry-after"] == str(api.PREDICTION_RETRY_AFTER_SECONDS)
    assert first_response.status_code == 200
    assert "retry-after" not in first_response.headers
    assert recovered_response.status_code == 200
    assert "retry-after" not in recovered_response.headers
    assert loaded_model.max_active_predictions == 1


def test_predict_retry_after_header_only_on_model_lane_overload(client, monkeypatch):
    make_ready(monkeypatch)
    assert api.PREDICTION_RETRY_AFTER_SECONDS == int(api.PREDICTION_QUEUE_TIMEOUT_SECONDS)

    success = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )
    validation = client.post("/predict")

    monkeypatch.setattr(api, "prediction_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(api, "PREDICTION_QUEUE_TIMEOUT_SECONDS", 0.01)
    overloaded = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert success.status_code == 200
    assert "retry-after" not in success.headers
    assert validation.status_code == 422
    assert "retry-after" not in validation.headers
    assert overloaded.status_code == 503
    assert overloaded.json() == {"detail": "Prediction queue is busy; retry later"}
    assert overloaded.headers["retry-after"] == str(api.PREDICTION_RETRY_AFTER_SECONDS)


def test_non_overload_unavailable_responses_omit_retry_after(client):
    health = client.get("/health")
    not_ready = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert health.status_code == 503
    assert "retry-after" not in health.headers
    assert not_ready.status_code == 503
    assert not_ready.json()["detail"] == "Model is not ready"
    assert "retry-after" not in not_ready.headers


def test_predict_bounds_image_processing_concurrency_and_recovers(monkeypatch):
    loaded_model = FakeModel([0.05, 0.1, 0.6, 0.15, 0.1])
    blocking_preprocessor = BlockingPreprocessor(expected_concurrency=2)
    monkeypatch.setattr(api, "load_model", lambda: loaded_model)
    monkeypatch.setattr(api, "load_class_mapping", mapping)
    monkeypatch.setattr(api, "preprocess_image", blocking_preprocessor)
    monkeypatch.setattr(api, "IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS", 0.05)

    def request_prediction(lifecycle_client):
        return lifecycle_client.post(
            "/predict",
            files={"image": ("car.png", b"image", "image/png")},
        )

    with TestClient(api.app) as lifecycle_client:
        with ThreadPoolExecutor(max_workers=4) as executor:
            first = executor.submit(request_prediction, lifecycle_client)
            second = executor.submit(request_prediction, lifecycle_client)
            assert blocking_preprocessor.expected_calls_started.wait(timeout=1)
            overloaded = executor.submit(request_prediction, lifecycle_client)
            health = executor.submit(lifecycle_client.get, "/health")

            try:
                overloaded_response = overloaded.result(timeout=1)
                health_while_blocked = health.result(timeout=1)
            finally:
                blocking_preprocessor.release_calls.set()

            active_responses = [first.result(timeout=2), second.result(timeout=2)]

        recovered_response = request_prediction(lifecycle_client)

    assert overloaded_response.status_code == 503
    assert overloaded_response.json() == {
        "detail": "Image processing queue is busy; retry later"
    }
    assert overloaded_response.headers["retry-after"] == "1"
    assert health_while_blocked.status_code == 200
    assert all(response.status_code == 200 for response in active_responses)
    assert recovered_response.status_code == 200
    assert blocking_preprocessor.max_active_calls == 2


def test_predict_rejects_non_finite_model_output_without_exposing_details(
    client, monkeypatch
):
    make_ready(monkeypatch)
    monkeypatch.setattr(
        api,
        "model",
        RawOutputModel(np.array([[0.1, 0.2, np.nan, 0.3, 0.4]])),
    )

    response = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"
    assert "finite" not in response.text


def test_predict_rejects_non_probability_output_without_exposing_details(
    client, monkeypatch
):
    make_ready(monkeypatch)
    monkeypatch.setattr(
        api,
        "model",
        RawOutputModel(np.array([[0.1, 0.2, 0.3, 0.2, 0.1]])),
    )

    response = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"
    assert "sum to one" not in response.text


def test_predict_does_not_expose_internal_errors(client, monkeypatch):
    make_ready(monkeypatch)
    monkeypatch.setattr(api, "model", FakeModel(error=RuntimeError("/private/model/path")))

    response = client.post(
        "/predict",
        files={"image": ("car.png", b"image", "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed"
    assert "/private/model/path" not in response.text


def _zero_metrics():
    snapshot = api.request_metrics.snapshot()
    assert snapshot["timings"] == {
        "preprocessing": {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        "inference_queue_wait": {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        "inference": {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0},
    }
    assert snapshot["rejections"] == {
        "unsupported_media_type": 0,
        "empty_image": 0,
        "oversized_upload": 0,
        "oversized_request": 0,
        "invalid_image": 0,
    }
    assert snapshot["unavailable"] == {
        "model_not_ready": 0,
        "image_processing_busy": 0,
        "prediction_queue_busy": 0,
    }


def test_metrics_aggregate_timings_rejections_and_unavailable(client, monkeypatch, caplog):
    caplog.set_level("INFO")
    sentinel_name = "secret-private-file.png"
    sentinel_bytes = b"SENTINEL_PAYLOAD_BYTES"
    upload = {"image": (sentinel_name, sentinel_bytes, "image/png")}

    health = client.get("/health")
    metrics = client.get("/metrics")

    assert health.status_code == 503
    assert metrics.status_code == 200
    _zero_metrics()
    not_ready = client.post("/predict", files=upload)
    assert not_ready.status_code == 503
    assert api.request_metrics.snapshot()["unavailable"]["model_not_ready"] == 1

    make_ready(monkeypatch)
    success = client.post("/predict", files=upload)
    unsupported = client.post(
        "/predict",
        files={"image": (sentinel_name, sentinel_bytes, "image/gif")},
    )
    empty = client.post(
        "/predict",
        files={"image": (sentinel_name, b"", "image/png")},
    )
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    oversized_upload = client.post("/predict", files=upload)
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)

    def reject_image(_data):
        raise ValueError("decoder internals")

    monkeypatch.setattr(api, "preprocess_image", reject_image)
    invalid = client.post("/predict", files=upload)
    monkeypatch.setattr(
        api,
        "preprocess_image",
        lambda _data: np.zeros((1, 224, 224, 3), dtype=np.float32),
    )
    monkeypatch.setattr(api, "image_processing_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(api, "IMAGE_PROCESSING_QUEUE_TIMEOUT_SECONDS", 0.01)
    image_busy = client.post("/predict", files=upload)
    monkeypatch.setattr(
        api,
        "image_processing_semaphore",
        asyncio.Semaphore(api.MAX_CONCURRENT_IMAGE_PROCESSING),
    )
    monkeypatch.setattr(api, "prediction_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(api, "PREDICTION_QUEUE_TIMEOUT_SECONDS", 0.01)
    prediction_busy = client.post("/predict", files=upload)
    monkeypatch.setattr(api, "MAX_REQUEST_BODY_BYTES", 4)
    oversized_request = client.post(
        "/predict",
        content=sentinel_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    published = client.get("/metrics")
    body = published.text

    assert success.status_code == 200
    assert unsupported.status_code == 400
    assert empty.status_code == 400
    assert oversized_upload.status_code == 413
    assert invalid.status_code == 400
    assert image_busy.status_code == 503
    assert prediction_busy.status_code == 503
    assert oversized_request.status_code == 413
    snapshot = published.json()
    assert snapshot["timings"]["preprocessing"]["count"] == 3
    assert snapshot["timings"]["inference_queue_wait"]["count"] == 2
    assert snapshot["timings"]["inference"]["count"] == 1
    assert snapshot["rejections"] == {
        "unsupported_media_type": 1,
        "empty_image": 1,
        "oversized_upload": 1,
        "oversized_request": 1,
        "invalid_image": 1,
    }
    assert snapshot["unavailable"] == {
        "model_not_ready": 1,
        "image_processing_busy": 1,
        "prediction_queue_busy": 1,
    }
    assert sentinel_name not in body
    assert "SENTINEL_PAYLOAD_BYTES" not in body
    assert sentinel_name not in caplog.text
    assert "SENTINEL_PAYLOAD_BYTES" not in caplog.text
    assert "/private/" not in body


def test_cancelled_queue_and_shutdown_do_not_leak_capacity(monkeypatch):
    loaded_model = BlockingModel([0.05, 0.1, 0.6, 0.15, 0.1])
    loaded_model.wait_timeout = 10
    monkeypatch.setattr(api, "load_model", lambda: loaded_model)
    monkeypatch.setattr(api, "load_class_mapping", mapping)
    monkeypatch.setattr(
        api,
        "preprocess_image",
        lambda _data: np.zeros((1, 224, 224, 3), dtype=np.float32),
    )
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "does not keep that permit" in readme
    assert "Shutdown waits for preprocessing or inference that already holds a lane" in readme
    assert "does not release the lane early" in readme
    assert "a later valid request can proceed" in readme

    async def post(client):
        return await client.post(
            "/predict",
            files={"image": ("car.png", b"image", "image/png")},
            timeout=10,
        )

    async def scenario():
        transport = httpx.ASGITransport(app=api.app)
        async with api.app.router.lifespan_context(api.app):
            semaphore = api.prediction_semaphore
            waiting = asyncio.Event()
            original_acquire = semaphore.acquire

            async def notifying_acquire():
                if semaphore.locked():
                    waiting.set()
                return await original_acquire()

            semaphore.acquire = notifying_acquire
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first = asyncio.create_task(post(client))
                assert await asyncio.to_thread(
                    loaded_model.first_prediction_started.wait, 2
                )
                queued = asyncio.create_task(post(client))
                await asyncio.wait_for(waiting.wait(), timeout=2)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
                assert loaded_model.active_predictions == 1
                assert loaded_model.max_active_predictions == 1
                assert semaphore._value == 0

                loaded_model.release_predictions.set()
                first_response = await first
                recovered = await post(client)

                assert first_response.status_code == 200
                assert recovered.status_code == 200
                assert semaphore._value == 1
                assert loaded_model.max_active_predictions == 1

        loaded_model.release_predictions.clear()
        loaded_model.first_prediction_started.clear()
        async with api.app.router.lifespan_context(api.app):
            semaphore = api.prediction_semaphore
            shutdown_event = api.shutting_down

            async def release_when_shutdown_starts():
                await shutdown_event.wait()
                assert loaded_model.active_predictions == 1
                assert semaphore._value == 0
                loaded_model.release_predictions.set()

            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                releaser = asyncio.create_task(release_when_shutdown_starts())
                owner = asyncio.create_task(post(client))
                assert await asyncio.to_thread(
                    loaded_model.first_prediction_started.wait, 2
                )
                assert loaded_model.active_predictions == 1
                assert semaphore._value == 0
                owner.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owner
                assert loaded_model.active_predictions == 1
                assert semaphore._value == 0
                monkeypatch.setattr(api, "PREDICTION_QUEUE_TIMEOUT_SECONDS", 0.05)
                busy = await post(client)
                assert busy.status_code == 503
                assert busy.json()["detail"] == api.DETAIL_PREDICTION_BUSY
                assert loaded_model.active_predictions == 1
                assert loaded_model.max_active_predictions == 1
            # Leaving the lifespan waits for the in-flight worker.
        await releaser
        assert loaded_model.active_predictions == 0
        assert semaphore._value == 1

        loaded_model.release_predictions.set()
        async with api.app.router.lifespan_context(api.app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                recovered_after_shutdown = await post(client)
        assert recovered_after_shutdown.status_code == 200
        assert loaded_model.max_active_predictions == 1

    asyncio.run(scenario())
