from pathlib import Path


def dependency_names(path):
    names = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            names.add(line.split("==", 1)[0].lower())
    return names


def test_api_requirements_include_runtime_contract_only():
    names = dependency_names("requirements-api.txt")

    assert {
        "tensorflow",
        "keras",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pillow",
        "numpy",
    }.issubset(names)
    assert names.isdisjoint(
        {
            "jupyter",
            "notebook",
            "jupyterlab",
            "matplotlib",
            "seaborn",
            "pandas",
            "kagglehub",
            "pytest",
        }
    )


def test_docker_uses_inference_requirements():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY requirements-api.txt ." in dockerfile
    assert "pip install --no-cache-dir -r requirements-api.txt" in dockerfile
    assert "COPY requirements.txt ." not in dockerfile
