"""Dependency-free model artifact discovery and integrity verification."""

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_CANDIDATES = (
    Path("best_car_model.keras"),
    Path("car_classification_model.h5"),
)
MODEL_MANIFEST = Path("model_manifest.json")
CLASS_MAPPING = Path("class_mapping.json")
SHA256_BUFFER_BYTES = 1024 * 1024


class ModelArtifactIntegrityError(RuntimeError):
    """The tracked manifest cannot authenticate its selected model artifact."""


def _validate_file_record(
    record: Any,
    *,
    label: str,
    allowed_paths: set[str],
) -> None:
    if not isinstance(record, dict):
        raise ModelArtifactIntegrityError(f"model manifest {label} record is invalid")

    relative_path = record.get("path")
    digest = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if relative_path not in allowed_paths:
        raise ModelArtifactIntegrityError(
            f"model manifest {label} path is not supported"
        )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ModelArtifactIntegrityError(
            f"model manifest {label} SHA-256 is invalid"
        )
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise ModelArtifactIntegrityError(f"model manifest {label} size is invalid")


def load_model_manifest(root: Path = Path(".")) -> dict[str, Any] | None:
    """Read and validate the tracked runtime selection when it is present."""
    root = Path(root)
    manifest_path = root / MODEL_MANIFEST
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactIntegrityError(
            f"cannot read {MODEL_MANIFEST}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ModelArtifactIntegrityError("model manifest schema is invalid")
    artifact = manifest.get("artifact")
    class_mapping = manifest.get("class_mapping")
    if manifest.get("schema_version") != 1:
        raise ModelArtifactIntegrityError("model manifest schema is invalid")
    _validate_file_record(
        artifact,
        label="artifact",
        allowed_paths={path.as_posix() for path in MODEL_CANDIDATES},
    )
    _validate_file_record(
        class_mapping,
        label="class mapping",
        allowed_paths={CLASS_MAPPING.as_posix()},
    )
    return manifest


def model_candidates(root: Path = Path(".")) -> tuple[Path, ...]:
    """Return the authoritative manifest selection or the legacy search order."""
    manifest = load_model_manifest(root)
    if manifest is not None:
        return (Path(manifest["artifact"]["path"]),)
    return MODEL_CANDIDATES


def _verify_file(path: Path, record: dict[str, Any], label: str) -> None:
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise ModelArtifactIntegrityError(f"cannot inspect {label}: {exc}") from exc
    if actual_size != record["size_bytes"]:
        raise ModelArtifactIntegrityError(
            f"{label} size does not match model_manifest.json"
        )

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(SHA256_BUFFER_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelArtifactIntegrityError(f"cannot hash {label}: {exc}") from exc
    if digest.hexdigest() != record["sha256"]:
        raise ModelArtifactIntegrityError(
            f"{label} SHA-256 does not match model_manifest.json"
        )


def verify_model_artifact(path: Path, root: Path = Path(".")) -> None:
    """Authenticate the selected weights and their exact class-index mapping."""
    root = Path(root)
    path = Path(path)
    manifest = load_model_manifest(root)
    if manifest is None:
        return

    artifact = manifest["artifact"]
    declared_path = root / artifact["path"]
    if path.resolve() != declared_path.resolve():
        raise ModelArtifactIntegrityError(
            "runtime model path does not match the authoritative model manifest"
        )
    _verify_file(path, artifact, "selected model artifact")

    mapping = manifest["class_mapping"]
    _verify_file(root / mapping["path"], mapping, "class mapping")


def find_model_artifact(root=Path(".")):
    """Return the first Keras 3 artifact supported by the API loader."""
    root = Path(root)
    for relative_path in model_candidates(root):
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None
