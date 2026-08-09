from pathlib import Path

import run


def test_model_discovery_matches_loader_preference(tmp_path):
    legacy = tmp_path / "car_classification_model.h5"
    legacy.touch()
    assert run.find_model_artifact(tmp_path) == legacy

    preferred = tmp_path / "best_car_model.keras"
    preferred.touch()
    assert run.find_model_artifact(tmp_path) == preferred


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
        "sudo docker build --build-arg MODEL_PATH=best_car_model.keras "
        "-t car-classification-service:latest ."
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
