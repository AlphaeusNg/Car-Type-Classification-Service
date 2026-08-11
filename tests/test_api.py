from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from io import BytesIO
from threading import Event, Lock

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
            if not self.release_predictions.wait(timeout=5):
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


def test_predict_rejects_unsupported_media_type(client, monkeypatch):
    make_ready(monkeypatch)

    response = client.post(
        "/predict",
        files={"image": ("car.gif", b"image", "image/gif")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File must be a JPEG or PNG image"


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
    assert overloaded_response.headers["retry-after"] == "5"
    assert first_response.status_code == 200
    assert recovered_response.status_code == 200
    assert loaded_model.max_active_predictions == 1


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
