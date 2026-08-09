from pathlib import Path
import subprocess

import argparse
import pytest

import run


def test_model_discovery_matches_loader_preference(tmp_path):
    legacy = tmp_path / "car_classification_model.h5"
    legacy.touch()
    assert run.find_model_artifact(tmp_path) == legacy

    preferred = tmp_path / "best_car_model.keras"
    preferred.touch()
    assert run.find_model_artifact(tmp_path) == preferred


@pytest.mark.parametrize("value", ["1", 8000, "65535"])
def test_valid_port_accepts_tcp_range(value):
    assert run.valid_port(value) == int(value)


@pytest.mark.parametrize("value", ["not-a-port", 0, -1, 65536])
def test_valid_port_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        run.valid_port(value)


def test_model_discovery_accepts_savedmodel_directory(tmp_path):
    saved_model = tmp_path / "models" / "car_classification_savedmodel"
    saved_model.mkdir(parents=True)

    assert run.find_model_artifact(tmp_path) == saved_model


def test_model_discovery_returns_none_when_artifact_is_missing(tmp_path):
    assert run.find_model_artifact(tmp_path) is None


def test_docker_build_uses_discovered_model_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("best_car_model.keras").touch()
    Path("class_mapping.json").touch()
    commands = []

    def capture(command, _description):
        commands.append(command)
        return True

    monkeypatch.setattr(run, "run_command", capture)

    assert run.build_docker()
    assert commands == [
        [
            "docker",
            "build",
            "--build-arg",
            "MODEL_PATH=best_car_model.keras",
            "-t",
            "car-classification-service:latest",
            ".",
        ]
    ]


def test_docker_build_fails_before_docker_when_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("class_mapping.json").touch()
    monkeypatch.setattr(
        run,
        "run_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not call Docker")),
    )

    assert not run.build_docker()


def test_run_command_passes_arguments_without_a_shell(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    assert run.run_command(["tool", "argument with spaces"], "Testing")
    assert calls[0][0] == ["tool", "argument with spaces"]
    assert "shell" not in calls[0][1]


def test_run_command_rejects_shell_strings():
    try:
        run.run_command("tool --flag", "Testing")
    except TypeError as error:
        assert "argument sequence" in str(error)
    else:
        raise AssertionError("shell string should be rejected")
