"""Integration tests running ncks (NCO) as a subprocess against the live server."""

import shutil
import subprocess

import pytest
import xarray as xr

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ncks") is None, reason="ncks not on PATH"),
]


class TestNcks:
    """Tests using ncks to query the remote OPeNDAP dataset."""

    def test_ncks_metadata(self, opendap_base):
        """ncks -m lists dimensions and variables."""
        result = subprocess.run(
            ["ncks", "-m", opendap_base],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"ncks failed: {result.stderr}"
        stdout = result.stdout
        assert "time" in stdout
        assert "lat" in stdout
        assert "lon" in stdout
        assert "air" in stdout

    def test_ncks_variable_extract(self, opendap_base, tmp_path):
        """ncks -v lat extracts latitude to a local file."""
        out_file = tmp_path / "lat.nc"
        result = subprocess.run(
            ["ncks", "-v", "lat", opendap_base, str(out_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"ncks failed: {result.stderr}"
        assert out_file.exists()
        ds = xr.open_dataset(out_file)
        assert "lat" in ds
        assert ds.sizes["lat"] == 25
        ds.close()

    def test_ncks_hyperslab(self, opendap_base, tmp_path):
        """ncks with -d flags extracts a single-point hyperslab."""
        out_file = tmp_path / "slice.nc"
        result = subprocess.run(
            [
                "ncks",
                "-d", "time,0,0",
                "-d", "lat,0,0",
                "-d", "lon,0,0",
                opendap_base,
                str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"ncks failed: {result.stderr}"
        assert out_file.exists()
        ds = xr.open_dataset(out_file)
        assert ds.sizes["time"] == 1
        assert ds.sizes["lat"] == 1
        assert ds.sizes["lon"] == 1
        ds.close()
