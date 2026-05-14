"""Verify the installed package exposes the correct API surface."""
import subprocess
import sys


def test_public_api():
    import autofacemonker
    assert callable(autofacemonker.AutoFaceMonker)
    assert callable(autofacemonker.register_template)
    assert isinstance(autofacemonker.__version__, str)


def test_version_format():
    import autofacemonker
    parts = autofacemonker.__version__.split(".")
    assert len(parts) == 3, "version must be X.Y.Z"
    assert all(p.isdigit() for p in parts)


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "autofacemonker._cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "target" in result.stdout
    assert "template" in result.stdout


def test_bundled_template_exists():
    from autofacemonker._register import _default_template
    import os
    path = _default_template()
    assert os.path.isfile(path), f"bundled template.ply not found at: {path}"
