import re
from pathlib import Path


WORKFLOW = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_workflow_policy():
    assert re.search(r"(?m)^name:\s*ci\s*$", WORKFLOW)
    assert re.search(r"push:\s*\n\s+branches:\s*\[main\]", WORKFLOW)
    assert re.search(r"(?m)^\s{2}pull_request:\s*$", WORKFLOW)
    assert re.search(r"permissions:\s*\n\s+contents:\s*read", WORKFLOW)
    assert re.search(
        r"concurrency:[\s\S]*group:\s*ci-.*github\.workflow.*github\.ref",
        WORKFLOW,
    )
    assert re.search(r"cancel-in-progress:\s*true", WORKFLOW)
    assert re.search(r"timeout-minutes:\s*10", WORKFLOW)
    assert re.search(r"uses:\s*actions/checkout@v7\b", WORKFLOW)
    assert re.search(r"uses:\s*actions/setup-python@v7\b", WORKFLOW)
    assert re.search(r"python-version:\s*[\"']3\.12[\"']", WORKFLOW)
    assert re.search(r"cache:\s*pip", WORKFLOW)
    assert "cache-dependency-path: requirements-test.txt" in WORKFLOW
    assert "python -m pip install -r requirements-test.txt" in WORKFLOW
    assert "python -m pytest -q tests/test_workflow.py" in WORKFLOW
    assert re.search(r"run:\s*python -m pytest -q\s*$", WORKFLOW, re.MULTILINE)
    assert "python -m pip check" in WORKFLOW
    assert (
        "python -m compileall -q api tests run.py prediction_example.py" in WORKFLOW
    )
    assert "actions/checkout@v4" not in WORKFLOW
    assert "actions/setup-python@v5" not in WORKFLOW
