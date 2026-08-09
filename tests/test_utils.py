import json
from pathlib import Path

import pytest

from api.utils import load_class_mapping, load_model, validate_image_dimensions


class LoadedModel:
    input_shape = (None, 224, 224, 3)


def test_image_dimensions_accept_decode_limit_boundary():
    validate_image_dimensions((10_000, 5_000))


@pytest.mark.parametrize("size", [(0, 100), (-1, 100), (10_001, 5_000)])
def test_image_dimensions_reject_invalid_or_excessive_sizes(size):
    with pytest.raises(ValueError):
        validate_image_dimensions(size)


def test_load_class_mapping_returns_valid_mapping(tmp_path):
    path = tmp_path / "mapping.json"
    expected = {
        "index_to_class": {"0": "coupe"},
        "class_to_index": {"coupe": 0},
    }
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_class_mapping(path) == expected


def test_load_class_mapping_reports_invalid_json_as_value_error(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_class_mapping(path)


def test_load_class_mapping_reports_invalid_structure_as_value_error(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"index_to_class": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Required keys"):
        load_class_mapping(path)


def test_load_class_mapping_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Class mapping not found"):
        load_class_mapping(tmp_path / "missing.json")


def test_load_model_uses_preferred_artifact_first(tmp_path):
    preferred = tmp_path / "best_car_model.keras"
    preferred.touch()
    calls = []

    def loader(path):
        calls.append(path)
        return LoadedModel()

    assert isinstance(load_model(tmp_path, loader), LoadedModel)
    assert calls == [str(preferred)]


def test_load_model_falls_back_after_preferred_artifact_fails(tmp_path):
    preferred = tmp_path / "best_car_model.keras"
    legacy = tmp_path / "car_classification_model.h5"
    preferred.touch()
    legacy.touch()
    calls = []

    def loader(path):
        calls.append(path)
        if path == str(preferred):
            raise ValueError("incompatible keras file")
        return LoadedModel()

    assert isinstance(load_model(tmp_path, loader), LoadedModel)
    assert calls == [str(preferred), str(legacy)]


def test_load_model_reports_existing_artifact_failures(tmp_path):
    preferred = tmp_path / "best_car_model.keras"
    legacy = tmp_path / "car_classification_model.h5"
    preferred.touch()
    legacy.touch()

    def loader(path):
        raise ValueError(f"cannot decode {Path(path).name}")

    with pytest.raises(RuntimeError, match="found but none could be loaded") as error:
        load_model(tmp_path, loader)

    message = str(error.value)
    assert "best_car_model.keras" in message
    assert "car_classification_model.h5" in message
    assert "cannot decode" in message


def test_load_model_reports_truly_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file found"):
        load_model(tmp_path, lambda _path: LoadedModel())
