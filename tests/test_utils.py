import json
from pathlib import Path

import numpy as np
import pytest

from api.utils import (
    decode_predictions,
    load_class_mapping,
    load_model,
    validate_image_dimensions,
)


class LoadedModel:
    input_shape = (None, 224, 224, 3)


def prediction_labels(size=3):
    return {str(index): f"class-{index}" for index in range(size)}


def test_decode_predictions_returns_ranked_finite_scores():
    decoded = decode_predictions(
        np.array([[0.1, 0.7, 0.2]], dtype=np.float32),
        prediction_labels(),
    )

    assert decoded["predicted_class"] == "class-1"
    assert decoded["confidence"] == pytest.approx(0.7)
    assert [item["class"] for item in decoded["top5_predictions"]] == [
        "class-1",
        "class-2",
        "class-0",
    ]


@pytest.mark.parametrize(
    ("predictions", "labels", "message"),
    [
        (np.array([0.1, 0.7, 0.2]), prediction_labels(), "2D"),
        (np.zeros((1, 1, 3)), prediction_labels(), "2D"),
        (np.zeros((2, 3)), prediction_labels(), "exactly one"),
        (np.empty((1, 0)), {}, "at least one"),
        (np.zeros((1, 2)), prediction_labels(), "width"),
        (np.zeros((1, 3)), {"0": "a", "1": "b", "3": "c"}, "contiguous"),
        (np.array([[0.1, np.nan, 0.2]]), prediction_labels(), "finite"),
        (np.array([[0.1, np.inf, 0.2]]), prediction_labels(), "finite"),
        (np.array([["low", "high", "mid"]]), prediction_labels(), "numeric"),
    ],
)
def test_decode_predictions_rejects_malformed_model_output(
    predictions, labels, message
):
    with pytest.raises(ValueError, match=message):
        decode_predictions(predictions, labels)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_decode_predictions_requires_positive_integer_top_k(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        decode_predictions(np.array([[0.1, 0.7, 0.2]]), prediction_labels(), top_k)


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
