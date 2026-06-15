"""Integration tests using Julia's NCDatasets.jl over DAP2."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("julia") is None, reason="julia not on PATH"),
]

_SCRIPT = str(Path(__file__).with_name("test_julia_client.jl"))


def _run_julia_test(opendap_url, test_name):
    """Run a single test case in the Julia script."""
    result = subprocess.run(
        ["julia", _SCRIPT, opendap_url, "--test", test_name],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Julia test {test_name!r} failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


class TestJuliaClient:
    """Tests using Julia NCDatasets.jl against the live OPeNDAP endpoint."""

    def test_open(self, opendap_base):
        _run_julia_test(opendap_base, "open")

    def test_dimensions(self, opendap_base):
        _run_julia_test(opendap_base, "dimensions")

    def test_variables(self, opendap_base):
        _run_julia_test(opendap_base, "variables")

    def test_variable_shape(self, opendap_base):
        _run_julia_test(opendap_base, "variable_shape")

    def test_read_lat(self, opendap_base):
        _run_julia_test(opendap_base, "read_lat")

    def test_read_data_slice(self, opendap_base):
        _run_julia_test(opendap_base, "read_data_slice")

    def test_attributes(self, opendap_base):
        _run_julia_test(opendap_base, "attributes")
