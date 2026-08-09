import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api


class FakeModel:
    def __init__(self, predictions=None, error=None):
        self.predictions = predictions
        self.error = error

    def predict(self, _image, verbose=0):
        assert verbose == 0
        if self.error:
            raise self.error
        return np.array([self.predictions], dtype=np.float32)


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
