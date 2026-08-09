import subprocess
import sys


def test_api_import_does_not_require_tensorflow():
    script = """
import sys

class RejectTensorFlow:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tensorflow" or fullname.startswith("tensorflow."):
            raise AssertionError("API contract import attempted to load TensorFlow")
        return None

sys.meta_path.insert(0, RejectTensorFlow())
import api.main
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
