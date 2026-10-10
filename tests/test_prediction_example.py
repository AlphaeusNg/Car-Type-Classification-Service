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

def test_default_prediction_uses_verified_repo_loader_from_any_directory(tmp_path, monkeypatch):
    import prediction_example
    from pathlib import Path

    image_path, _, mapping_path = write_fixture(tmp_path)
    calls = []
    loader = lambda _path, **_options: FakeModel([0.1, 0.7, 0.2])

    def verified_loader(*, project_root, model_loader):
        calls.append((project_root, model_loader))
        return FakeModel([0.1, 0.7, 0.2])

    monkeypatch.setattr(prediction_example, "load_model", verified_loader)
    monkeypatch.chdir(tmp_path)
    result = predict_car_type(image_path, mapping_path=mapping_path, model_loader=loader)
    assert result["predicted_class"] == "sedan"
    assert calls == [(Path(prediction_example.__file__).resolve().parent, loader)]



def test_explicit_prediction_rejects_tampered_manifest_artifact_before_loading(tmp_path):
    import hashlib
    from api.model_artifacts import ModelArtifactIntegrityError

    image_path, _, mapping_path = write_fixture(tmp_path)
    selected = tmp_path / "best_car_model.keras"
    trusted = b"trusted weights"
    selected.write_bytes(b"changed weights")
    mapping = mapping_path.read_bytes()
    (tmp_path / "class_mapping.json").write_bytes(mapping)
    (tmp_path / "model_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "artifact": {"path": selected.name, "size_bytes": len(trusted),
                     "sha256": hashlib.sha256(trusted).hexdigest()},
        "class_mapping": {"path": "class_mapping.json", "size_bytes": len(mapping),
                          "sha256": hashlib.sha256(mapping).hexdigest()},
    }), encoding="utf-8")
    calls = []
    with pytest.raises(ModelArtifactIntegrityError, match="SHA-256"):
        predict_car_type(image_path, selected, mapping_path,
                         model_loader=lambda *args, **options: calls.append(args))
    assert calls == []
