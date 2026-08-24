import re
from pathlib import Path


def test_inline_readme_python_paths_exist():
    readme = Path("README.md").read_text(encoding="utf-8")
    referenced = set(
        re.findall(r"`((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.py)`", readme)
    )
    missing = sorted(path for path in referenced if not Path(path).is_file())

    assert referenced, "README should name its Python entry points"
    assert not missing, f"README references missing Python files: {missing}"
