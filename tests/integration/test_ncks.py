"""Integration tests running ncks (NCO) as a subprocess against the live server."""

import shutil
import subprocess

import numpy as np
import pytest
import xarray as xr

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ncks") is None, reason="ncks not on PATH"),
]


class TestNcks:
    """Tests using ncks to query the remote OPeNDAP dataset."""

    def test_ncks_metadata(self, opendap_base):
        """Ncks -m lists dimensions and variables."""
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
        """Ncks -v lat extracts latitude to a local file."""
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
        """Ncks with -d flags extracts a single-point hyperslab."""
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


def _run_ncks(args, source, out_path):
    """Run ncks with the given args against a source, writing to out_path."""
    result = subprocess.run(
        ["ncks", *args, source, str(out_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"ncks failed: {result.stderr}"
    assert out_path.exists()
    return xr.open_dataset(out_path)


class TestNcksComparison:
    """Compare ncks output from a local netCDF file vs the OPeNDAP endpoint."""

    def test_variable_extract_match(self, reference_nc, opendap_base, tmp_path):
        """Ncks -v lat produces identical values from local file and OPeNDAP."""
        args = ["-v", "lat"]
        ds_local = _run_ncks(args, reference_nc, tmp_path / "local_lat.nc")
        ds_remote = _run_ncks(args, opendap_base, tmp_path / "opendap_lat.nc")

        assert dict(ds_local.sizes) == dict(ds_remote.sizes)
        np.testing.assert_allclose(
            ds_local["lat"].values,
            ds_remote["lat"].values,
        )
        ds_local.close()
        ds_remote.close()

    def test_hyperslab_match(self, reference_nc, opendap_base, tmp_path):
        """Single-point hyperslab produces identical values from both sources."""
        args = ["-d", "time,0,0", "-d", "lat,0,0", "-d", "lon,0,0"]
        ds_local = _run_ncks(args, reference_nc, tmp_path / "local_slice.nc")
        ds_remote = _run_ncks(args, opendap_base, tmp_path / "opendap_slice.nc")

        assert dict(ds_local.sizes) == dict(ds_remote.sizes)
        for var in ds_local.data_vars:
            np.testing.assert_allclose(
                ds_local[var].values,
                ds_remote[var].values,
                err_msg=f"Mismatch in variable {var!r}",
            )
        ds_local.close()
        ds_remote.close()

    def test_full_variable_data_match(self, reference_nc, opendap_base, tmp_path):
        """Full air variable data matches between local file and OPeNDAP."""
        args = ["-v", "air"]
        ds_local = _run_ncks(args, reference_nc, tmp_path / "local_air.nc")
        ds_remote = _run_ncks(args, opendap_base, tmp_path / "opendap_air.nc")

        assert dict(ds_local.sizes) == dict(ds_remote.sizes)
        np.testing.assert_allclose(
            ds_local["air"].values,
            ds_remote["air"].values,
        )
        ds_local.close()
        ds_remote.close()
