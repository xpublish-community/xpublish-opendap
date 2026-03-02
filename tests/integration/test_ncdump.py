"""Integration tests running ncdump as a subprocess against the live server."""

import shutil
import subprocess

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ncdump") is None, reason="ncdump not on PATH"),
]


class TestNcdump:
    """Tests using ncdump to inspect the remote OPeNDAP dataset."""

    def test_ncdump_header(self, opendap_base):
        """ncdump -h prints dimensions and variables."""
        result = subprocess.run(
            ["ncdump", "-h", opendap_base],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"ncdump failed: {result.stderr}"
        stdout = result.stdout
        assert "time = " in stdout
        assert "lat = " in stdout
        assert "lon = " in stdout
        assert "air" in stdout

    def test_ncdump_variable_slice(self, opendap_base):
        """ncdump -v lat prints latitude values."""
        result = subprocess.run(
            ["ncdump", "-v", "lat", opendap_base],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"ncdump failed: {result.stderr}"
        assert "lat" in result.stdout

    def test_ncdump_global_attrs(self, opendap_base):
        """ncdump -h includes global attributes."""
        result = subprocess.run(
            ["ncdump", "-h", opendap_base],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"ncdump failed: {result.stderr}"
        assert "title" in result.stdout.lower() or "global attributes" in result.stdout.lower()
