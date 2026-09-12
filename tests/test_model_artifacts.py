import hashlib
import json
from pathlib import Path

import pytest

from api.model_artifacts import (
    MODEL_MANIFEST,
    ModelArtifactIntegrityError,
    find_model_artifact,
    verify_model_artifact,
)


def write_manifest(root: Path, payload: bytes, **overrides) -> Path:
    artifact = {
        "path": "best_car_model.keras",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    artifact.update(overrides)
    mapping_payload = b"tracked class mapping"
    class_mapping = {
        "path": "class_mapping.json",
        "sha256": hashlib.sha256(mapping_payload).hexdigest(),
        "size_bytes": len(mapping_payload),
    }
    (root / MODEL_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": artifact,
                "class_mapping": class_mapping,
            }
        ),
        encoding="utf-8",
    )
    (root / class_mapping["path"]).write_bytes(mapping_payload)
    model_path = root / artifact["path"]
    model_path.write_bytes(payload)
    return model_path


def test_selected_artifact_matches_manifest(tmp_path):
    model_path = write_manifest(tmp_path, b"trusted model bytes")

    verify_model_artifact(model_path, tmp_path)


def test_manifest_is_optional_for_user_supplied_artifact(tmp_path):
    model_path = tmp_path / "best_car_model.keras"
    model_path.write_bytes(b"custom model")

    verify_model_artifact(model_path, tmp_path)


def test_manifest_rejects_a_different_runtime_model_path(tmp_path):
    write_manifest(tmp_path, b"selected model")
    legacy_path = tmp_path / "car_classification_model.h5"
    legacy_path.write_bytes(b"legacy model")

    with pytest.raises(ModelArtifactIntegrityError, match="authoritative"):
        verify_model_artifact(legacy_path, tmp_path)


def test_manifest_selected_h5_wins_even_when_preferred_name_exists(tmp_path):
    selected = write_manifest(
        tmp_path,
        b"selected legacy model",
        path="car_classification_model.h5",
    )
    (tmp_path / "best_car_model.keras").write_bytes(b"unselected model")

    assert find_model_artifact(tmp_path) == selected


def test_missing_manifest_selection_does_not_fall_back(tmp_path):
    selected = write_manifest(tmp_path, b"selected model")
    selected.unlink()
    (tmp_path / "car_classification_model.h5").write_bytes(b"legacy model")

    assert find_model_artifact(tmp_path) is None


def test_selected_artifact_size_mismatch_fails_closed(tmp_path):
    model_path = write_manifest(tmp_path, b"trusted model", size_bytes=999)

    with pytest.raises(ModelArtifactIntegrityError, match="size does not match"):
        verify_model_artifact(model_path, tmp_path)


def test_selected_artifact_digest_mismatch_fails_closed(tmp_path):
    model_path = write_manifest(tmp_path, b"trusted model", sha256="0" * 64)

    with pytest.raises(ModelArtifactIntegrityError, match="SHA-256 does not match"):
        verify_model_artifact(model_path, tmp_path)


def test_class_mapping_digest_mismatch_fails_closed(tmp_path):
    model_path = write_manifest(tmp_path, b"trusted model")
    (tmp_path / "class_mapping.json").write_bytes(b"x" * len(b"tracked class mapping"))

    with pytest.raises(ModelArtifactIntegrityError, match="class mapping SHA-256"):
        verify_model_artifact(model_path, tmp_path)


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"schema_version": 2, "artifact": {}},
        {
            "schema_version": 1,
            "artifact": {
                "path": "../model.keras",
                "sha256": "0" * 64,
                "size_bytes": 1,
            },
        },
        {
            "schema_version": 1,
            "artifact": {
                "path": "best_car_model.keras",
                "sha256": "not-a-digest",
                "size_bytes": 1,
            },
        },
    ],
)
def test_invalid_manifest_fails_closed(tmp_path, manifest):
    model_path = tmp_path / "best_car_model.keras"
    model_path.write_bytes(b"x")
    (tmp_path / MODEL_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactIntegrityError, match="manifest"):
        verify_model_artifact(model_path, tmp_path)


def test_committed_manifest_records_non_distributed_selected_artifact():
    manifest = json.loads(Path(MODEL_MANIFEST).read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {
        "path": "best_car_model.keras",
        "sha256": "a97b7d139d86c9f9fea7b9886e9bc73921e7528295a1644a3d342973f343029c",
        "size_bytes": 347612129,
    }
    assert manifest["class_mapping"] == {
        "path": "class_mapping.json",
        "sha256": "5fc7e7690897eed7a20fcd0971db84181bdf21a661179f353df6b1b7a1d74511",
        "size_bytes": 15838,
    }
    assert manifest["distribution"]["weights_tracked_in_git"] is False
