"""Integration tests using xr.open_dataset with various OPeNDAP engines."""

import sys

import numpy as np
import pytest
import xarray as xr

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="OPeNDAP engine tests fail on Windows CI",
    ),
]


class TestNetCDF4Engine:
    """Tests using xr.open_dataset(engine='netcdf4')."""

    @pytest.fixture(autouse=True)
    def _check_engine(self):
        pytest.importorskip("netCDF4")

    def test_open_netcdf4_engine(self, opendap_base):
        """Can open the dataset with the netcdf4 engine."""
        ds = xr.open_dataset(opendap_base, engine="netcdf4")
        assert ds is not None
        ds.close()

    def test_netcdf4_dimensions(self, opendap_base):
        """Dimension sizes match the reference."""
        ds = xr.open_dataset(opendap_base, engine="netcdf4")
        assert ds.sizes["time"] == 2920
        assert ds.sizes["lat"] == 25
        assert ds.sizes["lon"] == 53
        ds.close()

    def test_netcdf4_data_values(self, opendap_base, reference_ds):
        """A data slice matches the reference."""
        ds = xr.open_dataset(opendap_base, engine="netcdf4")
        actual = float(ds["air"].isel(time=0, lat=0, lon=0).values)
        expected = float(reference_ds["air"].isel(time=0, lat=0, lon=0).values)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        ds.close()

    def test_netcdf4_coordinates(self, opendap_base, reference_ds):
        """Coordinate arrays match the reference."""
        ds = xr.open_dataset(opendap_base, engine="netcdf4")
        np.testing.assert_allclose(
            ds["lat"].values,
            reference_ds["lat"].values,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            ds["lon"].values,
            reference_ds["lon"].values,
            rtol=1e-5,
        )
        ds.close()

    def test_netcdf4_attributes(self, opendap_base, reference_ds):
        """Key variable attributes are preserved."""
        ds = xr.open_dataset(opendap_base, engine="netcdf4")
        assert ds["air"].attrs["long_name"] == reference_ds["air"].attrs["long_name"]
        ds.close()


class TestPydapEngine:
    """Tests using xr.open_dataset(engine='pydap')."""

    @pytest.fixture(autouse=True)
    def _check_engine(self):
        pytest.importorskip("pydap")

    def test_open_pydap_engine(self, opendap_base):
        """Can open the dataset with the pydap engine."""
        ds = xr.open_dataset(opendap_base, engine="pydap")
        assert ds is not None
        ds.close()

    def test_pydap_data_values(self, opendap_base, reference_ds):
        """A data slice matches the reference via pydap engine."""
        ds = xr.open_dataset(opendap_base, engine="pydap")
        actual = float(ds["air"].isel(time=0, lat=0, lon=0).values)
        expected = float(reference_ds["air"].isel(time=0, lat=0, lon=0).values)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
        ds.close()
