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

def test_prediction_cli_missing_image_fails_without_loading_tensorflow(tmp_path):
    from pathlib import Path
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "prediction_example.py"),
                             str(tmp_path / "missing.png")], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 1
    assert "Image file does not exist" in result.stderr
    assert result.stdout == ""


def test_prediction_cli_reports_success_and_forwards_explicit_paths(tmp_path, monkeypatch, capsys):
    import prediction_example
    image, model, mapping = write_fixture(tmp_path)
    calls = []
    def predict(*args, **kwargs):
        calls.append(args)
        assert kwargs == {"top_k": 5}
        return {"predicted_class": "sedan", "confidence": 0.7,
                "top5_predictions": [{"class": "sedan", "confidence": 0.7}]}
    monkeypatch.setattr(prediction_example, "predict_car_type", predict)
    assert prediction_example.main([str(image), "--model", str(model), "--mapping", str(mapping)]) == 0
    assert calls == [(image, model, mapping)]
    assert "Class score: 70.00%" in capsys.readouterr().out

@pytest.mark.parametrize("contents", [b"not an image", b"GIF89a"])
def test_invalid_image_fails_before_model_loader(tmp_path, contents):
    image = tmp_path / "invalid.png"
    image.write_bytes(contents)
    calls = []
    with pytest.raises(ValueError, match="Failed to preprocess image"):
        predict_car_type(image, tmp_path / "missing.keras", model_loader=lambda *args, **kwargs: calls.append(args))
    assert calls == []

def test_invalid_image_cli_fails_before_tensorflow_import(tmp_path):
    from pathlib import Path
    image = tmp_path / "invalid.png"
    image.write_bytes(b"not an image")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "prediction_example.py"), str(image)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 1
    assert "Failed to preprocess image" in result.stderr
    assert "tensorflow" not in result.stderr.lower()
    assert result.stdout == ""


def test_json_cli_serializes_shared_top_k_predictions(tmp_path, monkeypatch, capsys):
    import prediction_example

    image, model, mapping = write_fixture(tmp_path)
    original = prediction_example.predict_car_type

    def loader(_path, **_options):
        print("loader progress")
        return FakeModel([0.4, 0.4, 0.2])

    monkeypatch.setattr(prediction_example, "predict_car_type", lambda *args, **kwargs:
                        original(*args, **kwargs, model_loader=loader))
    assert prediction_example.main([str(image), "--model", str(model),
                                    "--mapping", str(mapping), "--json", "--top-k", "2"]) == 0
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["schema_version"] == 1
    assert payload["status"] == "success"
    assert payload["class_index"] == 0
    assert payload["predicted_class"] == "coupe"
    assert [p["class"] for p in payload["top_predictions"]] == ["coupe", "sedan"]
    assert payload["confidence"] == pytest.approx(0.4)
    assert "loader progress" in output.err
    assert "Class mapping loaded" in output.err
    assert str(tmp_path) not in output.out


def test_prediction_top_k_larger_than_mapping_retains_python_contract(tmp_path):
    image, model, mapping = write_fixture(tmp_path)
    result = predict_car_type(image, model, mapping, top_k=196,
                              model_loader=lambda *_args, **_kwargs: FakeModel([0.4, 0.4, 0.2]))
    assert set(result) == {"predicted_class", "confidence", "top5_predictions", "class_index"}
    assert len(result["top5_predictions"]) == 3


@pytest.mark.parametrize("top_k", ["0", "197", "-1", "bad"])
def test_cli_rejects_top_k_before_predicting(tmp_path, monkeypatch, capsys, top_k):
    import prediction_example

    image, _, _ = write_fixture(tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("inference must not run")
    monkeypatch.setattr(prediction_example, "predict_car_type", forbidden)
    with pytest.raises(SystemExit) as error:
        prediction_example.main([str(image), "--json", "--top-k", top_k])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "--top-k" in output.err


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_json_cli_serialization_failure_writes_no_partial_document(tmp_path, monkeypatch, capsys, score):
    import prediction_example

    image, _, _ = write_fixture(tmp_path)
    monkeypatch.setattr(prediction_example, "predict_car_type", lambda *_args, **_kwargs:
                        {"predicted_class": "sedan", "class_index": 1,
                         "confidence": score, "top5_predictions": []})
    assert prediction_example.main([str(image), "--json"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err


def test_json_cli_subprocess_has_one_document_and_preserves_unicode(tmp_path):
    from pathlib import Path

    image, _, _ = write_fixture(tmp_path)
    script = """
import prediction_example
import sys
def predict(*args, **kwargs):
    print('runtime progress')
    return {'predicted_class': '轿车 café', 'class_index': 1, 'confidence': 0.7,
            'top5_predictions': [{'class': '轿车 café', 'confidence': 0.7}]}
prediction_example.predict_car_type = predict
raise SystemExit(prediction_example.main([sys.argv[1], '--json']))
"""
    result = subprocess.run([sys.executable, "-c", script, str(image)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True,
                            text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["predicted_class"] == "轿车 café"
    assert result.stdout.count("\n") == 1
    assert result.stderr == "runtime progress\n"


@pytest.mark.parametrize("failure", ["missing", "invalid"])
def test_json_cli_input_failure_has_empty_stdout(tmp_path, failure):
    from pathlib import Path

    image = tmp_path / "bad.png"
    if failure == "invalid":
        image.write_bytes(b"invalid image")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "prediction_example.py"),
                             str(image), "--json"], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "tensorflow" not in result.stderr.lower()


def test_json_cli_model_failure_redirects_diagnostics(tmp_path, monkeypatch, capsys):
    import prediction_example

    image, _, _ = write_fixture(tmp_path)
    def fail(*_args, **_kwargs):
        print("model progress")
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(prediction_example, "predict_car_type", fail)
    assert prediction_example.main([str(image), "--json"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "model progress\nError: model unavailable\n"
