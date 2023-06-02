"""Test OpenDAP clients against Xpublish OpenDAP plugin."""
import netCDF4
import pytest
import xarray as xr


def test_netcdf4(xpublish_server):
    """Test opening OpenDAP dataset directly with NetCDF4 library."""
    url = f"{xpublish_server}/datasets/air/opendap"
    netCDF4.Dataset(url)


def test_default_xarray_engine(xpublish_server, dataset):
    """Test opening OpenDAP dataset with default Xarray engine."""
    url = f"{xpublish_server}/datasets/air/opendap"
    ds = xr.open_dataset(url)
    assert ds == dataset


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf", "pydap"])
def test_xarray_engines(xpublish_server, engine, dataset):
    """Test opening OpenDAP dataset with specified engines."""
    url = f"{xpublish_server}/datasets/air/opendap"
    ds = xr.open_dataset(url, engine=engine)
    assert ds == dataset
