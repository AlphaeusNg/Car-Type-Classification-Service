import json

import pytest

from api.utils import load_class_mapping


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
