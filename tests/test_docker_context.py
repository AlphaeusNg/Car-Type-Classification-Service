from pathlib import Path


def test_docker_context_is_allowlisted_to_selected_runtime_inputs():
    rules = [
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert rules[0] == "**"
    assert {
        "!Dockerfile",
        "!requirements-api.txt",
        "!api/*.py",
        "!class_mapping.json",
        "!model_manifest.json",
        "!best_car_model.keras",
    }.issubset(rules)
    assert "!api/**" not in rules
    assert "!car_classification_model.h5" not in rules
    assert not any("savedmodel" in rule.lower() for rule in rules)


def test_runtime_image_drops_root_before_healthcheck_and_command():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "libgl1 \\" in dockerfile
    assert "libgl1-mesa-glx" not in dockerfile
    assert "useradd --system --gid app --create-home" in dockerfile
    assert "COPY --chown=app:app api/ api/" in dockerfile
    assert "COPY --chown=app:app model_manifest.json ." in dockerfile
    user_position = dockerfile.index("USER app")
    assert user_position < dockerfile.index("HEALTHCHECK")
    assert user_position < dockerfile.index('CMD ["uvicorn"')
