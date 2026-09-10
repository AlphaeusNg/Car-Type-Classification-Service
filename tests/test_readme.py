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


def test_docker_troubleshooting_avoids_host_wide_cleanup():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docker system prune" not in readme
    assert "docker logs car-classification-service" in readme
    assert "docker build --progress=plain --no-cache" in readme


def test_readme_license_matches_committed_license():
    readme = Path("README.md").read_text(encoding="utf-8")
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    license_heading = re.search(
        r"^\s*GNU GENERAL PUBLIC LICENSE\s*$",
        license_text,
        flags=re.MULTILINE,
    )
    license_version = re.search(r"^\s*Version (\d+),", license_text, flags=re.MULTILINE)
    readme_section = re.search(
        r"^## 📄 License\s*$(.*?)(?=^## |\Z)",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert license_heading, "LICENSE should identify the GNU GPL"
    assert license_version, "LICENSE should declare its major version"
    assert readme_section, "README should contain a License section"

    expected = f"GNU General Public License v{license_version.group(1)}.0"
    declaration = readme_section.group(1)
    assert expected in declaration
    assert "MIT License" not in declaration
