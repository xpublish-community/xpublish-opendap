"""Integration tests using netCDF4.Dataset over DAP2."""

import sys

import numpy as np
import pytest

try:
    import netCDF4

    HAS_NETCDF4 = True
except ImportError:
    HAS_NETCDF4 = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not installed"),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="netCDF4 OPeNDAP tests fail on Windows CI",
    ),
]


def _open(url):
    return netCDF4.Dataset(url)


class TestNetCDF4Client:
    """Tests using netCDF4-python's built-in DAP2 client."""

    def test_open_dataset(self, opendap_base):
        """Can open the remote dataset without errors."""
        ds = _open(opendap_base)
        ds.close()

    def test_dimensions(self, opendap_base):
        """Dimension names and sizes match the reference dataset."""
        ds = _open(opendap_base)
        assert len(ds.dimensions["time"]) == 2920
        assert len(ds.dimensions["lat"]) == 25
        assert len(ds.dimensions["lon"]) == 53
        ds.close()

    def test_variable_names(self, opendap_base):
        """Expected variables are present."""
        ds = _open(opendap_base)
        assert "air" in ds.variables
        assert "lat" in ds.variables
        assert "lon" in ds.variables
        assert "time" in ds.variables
        ds.close()

    def test_variable_shapes(self, opendap_base):
        """Air variable has the expected shape."""
        ds = _open(opendap_base)
        assert ds.variables["air"].shape == (2920, 25, 53)
        ds.close()

    def test_read_coordinate_values(self, opendap_base, reference_ds):
        """Latitude values match the reference dataset."""
        ds = _open(opendap_base)
        np.testing.assert_allclose(
            ds.variables["lat"][:],
            reference_ds["lat"].values,
            rtol=1e-5,
        )
        ds.close()

    def test_read_data_slice(self, opendap_base, reference_ds):
        """A single data value matches the reference."""
        ds = _open(opendap_base)
        actual = float(ds.variables["air"][0, 0, 0])
        expected = float(reference_ds["air"].values[0, 0, 0])
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        ds.close()

    def test_read_attributes(self, opendap_base, reference_ds):
        """Variable attributes are readable."""
        ds = _open(opendap_base)
        assert ds.variables["air"].long_name == reference_ds["air"].attrs["long_name"]
        ds.close()

    def test_global_attributes(self, opendap_base):
        """Global attributes are present."""
        ds = _open(opendap_base)
        assert hasattr(ds, "title") or "title" in ds.ncattrs()
        ds.close()
