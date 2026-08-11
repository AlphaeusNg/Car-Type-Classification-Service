from pathlib import Path


def dependency_names(path):
    return set(dependency_versions(path))


def dependency_versions(path):
    versions = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            versions[name.lower()] = version
    return versions


def test_api_requirements_include_runtime_contract_only():
    names = dependency_names("requirements-api.txt")

    assert {
        "tensorflow",
        "keras",
        "fastapi",
        "starlette",
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


def test_audited_runtime_and_test_pins_stay_aligned():
    workspace = dependency_versions("requirements.txt")
    api = dependency_versions("requirements-api.txt")
    tests = dependency_versions("requirements-test.txt")

    expected = {
        "fastapi": "0.139.2",
        "starlette": "1.6.0",
        "python-multipart": "0.0.32",
        "pillow": "12.3.0",
        "keras": "3.10.0",
        "pytest": "9.1.1",
    }
    for name, version in expected.items():
        assert workspace[name] == version
        if name != "pytest":
            assert api[name] == version
        if name != "keras":
            assert tests[name] == version
    assert tests["httpx2"] == "2.7.0"
    assert "httpx" not in tests


def test_docker_uses_inference_requirements():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY requirements-api.txt ." in dockerfile
    assert "pip install --no-cache-dir -r requirements-api.txt" in dockerfile
    assert "COPY requirements.txt ." not in dockerfile
