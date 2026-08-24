import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from prediction_example import predict_car_type


class FakeModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.input_shape = (None, 224, 224, 3)
        self.output_shape = (None, len(predictions))
        self.received = None

    def predict(self, image, verbose=0):
        assert verbose == 0
        self.received = image
        return np.array([self.predictions], dtype=np.float32)


def write_fixture(tmp_path):
    image_path = tmp_path / "car.png"
    Image.new("RGB", (32, 16), "red").save(image_path, format="PNG")
    model_path = tmp_path / "car.keras"
    model_path.touch()
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "index_to_class": {"0": "coupe", "1": "sedan", "2": "wagon"},
                "class_to_index": {"coupe": 0, "sedan": 1, "wagon": 2},
            }
        ),
        encoding="utf-8",
    )
    return image_path, model_path, mapping_path


def test_prediction_example_import_stays_lightweight():
    script = """
import sys

class RejectTensorFlow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorflow" or fullname.startswith("tensorflow."):
            raise AssertionError("prediction example imported TensorFlow eagerly")
        return None

sys.meta_path.insert(0, RejectTensorFlow())
import prediction_example
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_prediction_example_uses_shared_safe_inference_contract(tmp_path):
    image_path, model_path, mapping_path = write_fixture(tmp_path)
    loaded = FakeModel([0.1, 0.7, 0.2])
    calls = []

    def loader(path, **options):
        calls.append((path, options))
        return loaded

    result = predict_car_type(
        image_path,
        model_path,
        mapping_path,
        model_loader=loader,
    )

    assert calls == [(str(model_path), {"compile": False})]
    assert loaded.received.shape == (1, 224, 224, 3)
    assert loaded.received.dtype == np.float32
    assert result["predicted_class"] == "sedan"
    assert result["class_index"] == 1
    assert [item["class"] for item in result["top5_predictions"]] == [
        "sedan",
        "wagon",
        "coupe",
    ]


def test_prediction_example_rejects_non_probability_outputs(tmp_path):
    image_path, model_path, mapping_path = write_fixture(tmp_path)

    with pytest.raises(ValueError, match="sum to one"):
        predict_car_type(
            image_path,
            model_path,
            mapping_path,
            model_loader=lambda _path, **_options: FakeModel([0.1, 0.2, 0.3]),
        )
