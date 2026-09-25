#!/usr/bin/env python3
"""Compare a candidate Keras re-export with the trusted serving artifact.

The trusted artifact is the file named by ``model_manifest.json``. Loading it
requires Keras 3.10.0. This tool only reads that artifact, the class mapping,
and a fixed relative sample of test images. It does not replace
``best_car_model.keras``, ``model_manifest.json``, or any other serving file.

TensorFlow and Keras are imported only after ``--candidate`` is parsed, so
``--help`` stays lightweight. If they cannot be imported, or Keras is not
3.10.0, the process fails closed and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.model_artifacts import (  # noqa: E402
    ModelArtifactIntegrityError,
    load_model_manifest,
    verify_model_artifact,
)
from api.utils import preprocess_image  # noqa: E402
REQUIRED_KERAS_VERSION = "3.10.0"
PROTECTED_OUTPUTS = (
    Path("best_car_model.keras"),
    Path("model_manifest.json"),
    Path("class_mapping.json"),
)
# Fixed spread across the committed label order. Not the official 8,041-image test.
REPRESENTATIVE_SAMPLES = (
    "data/test/AM General Hummer SUV 2000/000046.jpg",
    "data/test/Acura TL Sedan 2012/000197.jpg",
    "data/test/BMW M3 Coupe 2012/002744.jpg",
    "data/test/Chevrolet Corvette Convertible 2012/004425.jpg",
    "data/test/Ford F-150 Regular Cab 2012/009243.jpg",
    "data/test/Jeep Wrangler SUV 2012/011981.jpg",
    "data/test/Tesla Model S Sedan 2012/015213.jpg",
    "data/test/smart fortwo Convertible 2012/016146.jpg",
)


@dataclass(frozen=True)
class ScoreTolerances:
    """Absolute score tolerances for one re-export comparison."""

    max_abs_score: float = 1e-3
    mean_abs_score: float = 1e-4
    require_same_argmax: bool = True


TOLERANCES = ScoreTolerances()


def fail_closed(message: str) -> NoReturn:
    """Stop before any serving file can be changed."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def require_supported_keras(version: str | None) -> None:
    """Refuse every Keras release that is not the artifact's known loader."""
    if version != REQUIRED_KERAS_VERSION:
        fail_closed(
            f"Keras {version} cannot load the trusted artifact. "
            f"Keras {REQUIRED_KERAS_VERSION} is required. Refusing to continue."
        )


def is_protected_output(path: Path, root: Path = ROOT) -> bool:
    """True when a path is a serving artifact this tool must not replace."""
    resolved_root = root.resolve()
    try:
        candidate = path.resolve()
    except OSError:
        return False
    protected = {(resolved_root / relative).resolve() for relative in PROTECTED_OUTPUTS}
    return candidate in protected


def load_inference_stack() -> tuple[Any, Any]:
    """Import TensorFlow and Keras, or fail closed before touching weights."""
    try:
        import keras
        import tensorflow as tensorflow_module
    except ImportError:
        fail_closed(
            "TensorFlow/Keras is not importable. This equivalence check "
            f"requires Keras {REQUIRED_KERAS_VERSION} to load the trusted "
            ".keras artifact and refuses to continue without it."
        )
    require_supported_keras(getattr(keras, "__version__", None))
    tf_keras = getattr(tensorflow_module, "keras", None)
    require_supported_keras(getattr(tf_keras, "__version__", None))
    return tensorflow_module, keras


def trusted_reference(root: Path) -> Path:
    """Return the manifest-selected artifact without modifying the manifest."""
    manifest = load_model_manifest(root)
    if manifest is None:
        fail_closed(
            "model_manifest.json is required to identify the trusted artifact. "
            "Refusing to continue."
        )
    return root / manifest["artifact"]["path"]


def compare_score_rows(
    reference_scores: Any,
    candidate_scores: Any,
    tolerances: ScoreTolerances = TOLERANCES,
) -> dict[str, Any]:
    """Compare one reference score row with one candidate score row."""
    import numpy as np

    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if reference.ndim == 2 and reference.shape[0] == 1:
        reference = reference[0]
    if candidate.ndim == 2 and candidate.shape[0] == 1:
        candidate = candidate[0]
    if reference.ndim != 1 or candidate.ndim != 1 or reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": "score shape mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"passed": False, "reason": "non-finite scores"}

    delta = np.abs(reference - candidate)
    max_abs = float(delta.max()) if delta.size else 0.0
    mean_abs = float(delta.mean()) if delta.size else 0.0
    reference_argmax = int(np.argmax(reference))
    candidate_argmax = int(np.argmax(candidate))
    passed = max_abs <= tolerances.max_abs_score and mean_abs <= tolerances.mean_abs_score
    if tolerances.require_same_argmax:
        passed = passed and reference_argmax == candidate_argmax
    return {
        "passed": passed,
        "max_abs_score": max_abs,
        "mean_abs_score": mean_abs,
        "argmax_reference": reference_argmax,
        "argmax_candidate": candidate_argmax,
        "max_abs_tolerance": tolerances.max_abs_score,
        "mean_abs_tolerance": tolerances.mean_abs_score,
    }


def compare_loaded_models(reference: Any, candidate: Any) -> dict[str, Any]:
    """Compare load-time contracts. This does not write either model."""
    reference_params = int(reference.count_params())
    candidate_params = int(candidate.count_params())
    passed = (
        tuple(reference.input_shape) == tuple(candidate.input_shape)
        and tuple(reference.output_shape) == tuple(candidate.output_shape)
        and reference_params == candidate_params
    )
    return {
        "passed": passed,
        "reference_input_shape": list(reference.input_shape),
        "candidate_input_shape": list(candidate.input_shape),
        "reference_output_shape": list(reference.output_shape),
        "candidate_output_shape": list(candidate.output_shape),
        "reference_parameters": reference_params,
        "candidate_parameters": candidate_params,
    }


def missing_samples(root: Path) -> list[str]:
    """Return declared relative samples that are not files."""
    missing = []
    for relative in REPRESENTATIVE_SAMPLES:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            missing.append(relative)
    return missing


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        help=(
            "candidate .keras re-export to compare. "
            "This path is read only and is never copied onto the serving artifact."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="repository root containing the trusted manifest (read only)",
    )
    return parser.parse_args(argv)


def load_model_read_only(loader: Any, path: Path) -> Any:
    """Load one artifact with compile=False. The file is not rewritten."""
    return loader(str(path), compile=False)


def evaluate_candidate(root: Path, candidate_path: Path, loader: Any) -> dict[str, Any]:
    """Load both artifacts and compare the declared sample. Writes nothing."""
    if is_protected_output(candidate_path, root) and candidate_path.resolve() != (
        root / "best_car_model.keras"
    ).resolve():
        fail_closed(
            "Refusing to use a protected serving path as the candidate. "
            "This tool does not replace best_car_model.keras or model_manifest.json."
        )
    reference_path = trusted_reference(root)
    try:
        verify_model_artifact(reference_path, root)
    except ModelArtifactIntegrityError as exc:
        fail_closed(
            "Trusted artifact failed manifest verification. "
            f"Refusing to continue: {exc}"
        )
    if not candidate_path.is_file():
        fail_closed("Candidate artifact is missing. Refusing to continue.")
    missing = missing_samples(root)
    if missing:
        fail_closed(
            "Representative sample images are missing: " + ", ".join(missing)
        )

    reference_model = load_model_read_only(loader, reference_path)
    candidate_model = load_model_read_only(loader, candidate_path)
    loading = compare_loaded_models(reference_model, candidate_model)
    samples = []
    for relative in REPRESENTATIVE_SAMPLES:
        try:
            batch = preprocess_image((root / relative).read_bytes())
            reference_scores = reference_model.predict(batch, verbose=0)
            candidate_scores = candidate_model.predict(batch, verbose=0)
        except Exception as exc:
            fail_closed(
                "Could not score representative sample "
                f"{relative} ({type(exc).__name__})."
            )
        comparison = compare_score_rows(reference_scores, candidate_scores)
        comparison["sample"] = relative
        samples.append(comparison)
    passed = loading["passed"] and all(sample["passed"] for sample in samples)
    return {
        "promoted": False,
        "passed": passed,
        "keras": REQUIRED_KERAS_VERSION,
        "tolerances": asdict(TOLERANCES),
        "loading": loading,
        "samples": samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    candidate_path = Path(args.candidate)
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    tensorflow_module, _keras = load_inference_stack()
    report = evaluate_candidate(
        root,
        candidate_path,
        tensorflow_module.keras.models.load_model,
    )
    import json

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
