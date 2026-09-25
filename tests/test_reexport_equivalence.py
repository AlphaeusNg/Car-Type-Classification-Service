import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tools import check_reexport_equivalence as tool


def test_equivalence_tool_import_does_not_load_tensorflow():
    script = """
import sys

class RejectTensorFlow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorflow" or fullname.startswith("tensorflow."):
            raise AssertionError("equivalence tool imported TensorFlow eagerly")
        if fullname == "keras" or fullname.startswith("keras."):
            raise AssertionError("equivalence tool imported Keras eagerly")
        return None

sys.meta_path.insert(0, RejectTensorFlow())
import tools.check_reexport_equivalence
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_equivalence_help_stays_lightweight():
    result = subprocess.run(
        [sys.executable, "tools/check_reexport_equivalence.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Keras 3.10.0" in result.stdout
    assert "best_car_model.keras" in result.stdout


def test_equivalence_tool_fails_closed_without_tensorflow():
    root = Path("best_car_model.keras")
    manifest = Path("model_manifest.json")
    before = (root.stat().st_mtime_ns, root.stat().st_size) if root.exists() else None
    manifest_before = (manifest.stat().st_mtime_ns, manifest.stat().st_size)
    script = """
import sys

class RejectTensorFlow:
    def find_spec(self, fullname, path=None, target=None):
        blocked = fullname == "tensorflow" or fullname.startswith("tensorflow.")
        blocked = blocked or fullname == "keras" or fullname.startswith("keras.")
        if blocked:
            raise ImportError(fullname)
        return None

sys.meta_path.insert(0, RejectTensorFlow())
import runpy
sys.argv = ["check_reexport_equivalence.py", "--candidate", "missing-candidate.keras"]
runpy.run_path("tools/check_reexport_equivalence.py", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "not importable" in result.stderr
    assert "Keras 3.10.0" in result.stderr
    assert manifest_before == (manifest.stat().st_mtime_ns, manifest.stat().st_size)
    if before is not None:
        assert before == (root.stat().st_mtime_ns, root.stat().st_size)


def test_keras_version_gate_rejects_untested_releases():
    tool.require_supported_keras("3.10.0")
    with pytest.raises(SystemExit) as blocked:
        tool.require_supported_keras("3.11.3")

    assert blocked.value.code == 2


def test_score_and_load_comparisons_use_declared_tolerances():
    assert tool.TOLERANCES.max_abs_score == 1e-3
    assert tool.TOLERANCES.mean_abs_score == 1e-4
    assert tool.TOLERANCES.require_same_argmax is True

    same = tool.compare_score_rows([0.25, 0.75], [[0.25, 0.75]])
    swapped = tool.compare_score_rows([0.25, 0.75], [0.75, 0.25])
    shifted = tool.compare_score_rows([0.25, 0.75], [0.25, 0.75 + 0.02])

    assert same["passed"] is True
    assert same["argmax_reference"] == 1
    assert swapped["passed"] is False
    assert shifted["passed"] is False
    assert shifted["max_abs_score"] == pytest.approx(0.02)

    class Loaded:
        def __init__(self, input_shape, output_shape, params):
            self.input_shape = input_shape
            self.output_shape = output_shape
            self._params = params

        def count_params(self):
            return self._params

    reference = Loaded((None, 224, 224, 3), (None, 196), 10)
    assert tool.compare_loaded_models(reference, reference)["passed"] is True
    assert tool.compare_loaded_models(
        reference, Loaded((None, 224, 224, 3), (None, 5), 10)
    )["passed"] is False


def test_representative_sample_stays_relative_and_small():
    assert 1 <= len(tool.REPRESENTATIVE_SAMPLES) <= 16
    for relative in tool.REPRESENTATIVE_SAMPLES:
        path = Path(relative)
        assert path.is_absolute() is False
        assert ".." not in path.parts
        assert path.parts[:2] == ("data", "test")


def test_protected_outputs_are_the_serving_files(tmp_path):
    artifact = tmp_path / "best_car_model.keras"
    manifest = tmp_path / "model_manifest.json"
    mapping = tmp_path / "class_mapping.json"
    report = tmp_path / "equivalence-report.json"
    artifact.write_bytes(b"weights")
    manifest.write_text("{}", encoding="utf-8")
    mapping.write_text("{}", encoding="utf-8")

    assert tool.is_protected_output(artifact, tmp_path)
    assert tool.is_protected_output(manifest, tmp_path)
    assert tool.is_protected_output(mapping, tmp_path)
    assert tool.is_protected_output(report, tmp_path) is False


def test_equivalence_tool_has_no_promotion_writes():
    source = Path("tools/check_reexport_equivalence.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"write_text", "write_bytes", "copy", "copy2", "copyfile", "move", "replace"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"copy", "copyfile", "move"}
    assert "promoted" in source
    assert "compile=False" in source
