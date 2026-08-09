from pathlib import Path


def test_docker_context_is_allowlisted_and_keeps_supported_artifacts():
    rules = [
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert rules[0] == "**"
    assert {
        "!Dockerfile",
        "!requirements.txt",
        "!api/**",
        "!class_mapping.json",
        "!best_car_model.keras",
        "!car_classification_model.h5",
        "!models/car_classification_savedmodel/**",
    }.issubset(rules)


def test_runtime_image_drops_root_before_healthcheck_and_command():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "useradd --system --gid app --create-home" in dockerfile
    assert "COPY --chown=app:app api/ api/" in dockerfile
    user_position = dockerfile.index("USER app")
    assert user_position < dockerfile.index("HEALTHCHECK")
    assert user_position < dockerfile.index('CMD ["uvicorn"')
