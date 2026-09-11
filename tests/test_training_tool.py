import ast
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_NAME = "tools.train_optimized"


def load_training_tool():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_training_tool_import_does_not_load_tensorflow():
    tensorflow_was_loaded = "tensorflow" in sys.modules

    tool = load_training_tool()

    assert tool.NUM_CLASSES == 196
    assert ("tensorflow" in sys.modules) is tensorflow_was_loaded


def test_training_cli_requires_an_explicit_operation():
    tool = load_training_tool()

    with pytest.raises(SystemExit) as missing:
        tool.parse_args([])
    with pytest.raises(SystemExit) as conflicting:
        tool.parse_args(["--train", "--evaluate-run", "candidate"])

    assert missing.value.code == 2
    assert conflicting.value.code == 2
    assert tool.parse_args(["--train", "--run-name", "candidate"]).train is True


@pytest.mark.parametrize(
    "name",
    ["../outside", "nested/run", ".", "", "run name", "a" * 65],
)
def test_run_names_cannot_escape_or_alias_the_bounded_run_root(name):
    tool = load_training_tool()

    with pytest.raises(ValueError, match="run name"):
        tool.resolve_run_paths(name)


def test_run_outputs_are_scoped_to_one_named_ignored_directory():
    tool = load_training_tool()

    paths = tool.resolve_run_paths("cycle-42")

    assert paths.directory == tool.RUNS_DIR / "cycle-42"
    assert paths.checkpoint.parent == paths.directory
    assert paths.history.parent == paths.directory
    assert paths.metrics.parent == paths.directory
    assert paths.status.parent == paths.directory
    assert paths.epoch_log.parent == paths.directory


def test_successful_status_clears_a_stale_failure(tmp_path):
    tool = load_training_tool()
    paths = SimpleNamespace(status=tmp_path / "status.json")

    tool.write_status(paths, state="failed", error="first attempt failed")
    tool.write_status(paths, state="completed", metrics={"accuracy": 0.5})

    status = json.loads(paths.status.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert "error" not in status


def test_model_uses_exactly_one_efficientnet_preprocessing_contract():
    source = Path("tools/train_optimized.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    efficientnet_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "EfficientNetV2S"
    ]

    assert len(efficientnet_calls) == 1
    keyword = next(
        item
        for item in efficientnet_calls[0].keywords
        if item.arg == "include_preprocessing"
    )
    assert isinstance(keyword.value, ast.Constant)
    assert keyword.value.value is False


def test_candidate_metrics_mark_test_as_report_only():
    tool = load_training_tool()

    payload = tool.build_metrics(
        val_results={"accuracy": 0.7},
        test_results={"accuracy": 0.6},
        history={"accuracy": [0.5, 0.7]},
    )

    assert payload["selection_metric"] == "val_accuracy"
    assert payload["test_role"] == "final_report_only"
    assert "promoted" not in payload


def test_cpu_execution_requires_opt_in_and_forces_float32():
    tool = load_training_tool()
    selected_policies = []
    tool.tf = SimpleNamespace(
        config=SimpleNamespace(list_physical_devices=lambda kind: [])
    )
    tool.keras = SimpleNamespace(
        mixed_precision=SimpleNamespace(set_global_policy=selected_policies.append)
    )

    with pytest.raises(RuntimeError, match="--allow-cpu"):
        tool.configure_runtime(allow_cpu=False)

    assert tool.configure_runtime(allow_cpu=True) == "cpu_float32"
    assert selected_policies == ["float32"]


def test_gpu_execution_enables_memory_growth_and_mixed_precision():
    tool = load_training_tool()
    gpu = object()
    memory_growth_calls = []
    selected_policies = []
    tool.tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda kind: [gpu],
            experimental=SimpleNamespace(
                set_memory_growth=lambda device, enabled: memory_growth_calls.append(
                    (device, enabled)
                )
            ),
        )
    )
    tool.keras = SimpleNamespace(
        mixed_precision=SimpleNamespace(set_global_policy=selected_policies.append)
    )

    assert tool.configure_runtime(allow_cpu=False) == "gpu_mixed_float16"
    assert memory_growth_calls == [(gpu, True)]
    assert selected_policies == ["mixed_float16"]


def test_training_tool_has_no_deployed_model_write_target():
    source = Path("tools/train_optimized.py").read_text(encoding="utf-8")

    assert "PROMOTED_PATH" not in source
    assert "best_car_model.previous.keras" not in source
    assert "shutil.copy" not in source
