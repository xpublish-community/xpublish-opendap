"""Test OpenDAP clients against Xpublish OpenDAP plugin.

Live tests are currently failing on Windows, see:
- https://github.com/Unidata/netcdf-c/issues/2459
- https://github.com/Unidata/netcdf4-python/issues/1246
- https://github.com/pydata/xarray/issues/7773
"""

import http.client
import sys

import netCDF4
import pytest
import xarray as xr


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
def test_netcdf4(xpublish_server):
    """Test opening OpenDAP air dataset directly with NetCDF4 library."""
    url = f"{xpublish_server}/datasets/air/opendap"
    netCDF4.Dataset(url)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
def test_default_xarray_engine(xpublish_server, dataset):
    """Test opening OpenDAP air dataset with default Xarray engine."""
    url = f"{xpublish_server}/datasets/air/opendap"
    ds = xr.open_dataset(url)
    xr.testing.assert_equal(ds, dataset)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
@pytest.mark.parametrize(
    "engine",
    [
        "netcdf4",
        pytest.param(
            "h5netcdf",
            marks=pytest.mark.xfail(
                reason="fails with 404 not found",
                raises=FileNotFoundError,
                strict=True,
            ),
        ),
        pytest.param(
            "pydap",
            marks=pytest.mark.xfail(
                reason="Incomplete Read",
                raises=http.client.IncompleteRead,
                strict=True,
            ),
        ),
    ],
)
def test_xarray_engines(xpublish_server, engine, dataset):
    """Test opening OpenDAP dataset with specified engines."""
    url = f"{xpublish_server}/datasets/air/opendap"
    ds = xr.open_dataset(url, engine=engine)
    xr.testing.assert_equal(ds, dataset)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
def test_attrs_quotes(xpublish_server):
    """Test that we are formatting OpenDAP attributes that contain '"' properly."""
    url = f"{xpublish_server}/datasets/attrs_quote/opendap"
    ds = xr.open_dataset(url)

    assert ds.attrs["quotes"] == 'This attribute uses "quotes"'


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
def test_attrs_types(xpublish_server):
    """Test that we are formatting OpenDAP attributes that contain '"' properly."""
    url = f"{xpublish_server}/datasets/attrs_cast/opendap"
    ds = xr.open_dataset(url)

    assert ds.attrs["npint"] == 16
    assert ds.attrs["npintthirtytwo"] == 32


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
@pytest.mark.xfail(
    raises=IndexError,
    reason="Unable to decode ocean_time",
    strict=True,
)
def test_coordinates_persist_correctly(xpublish_server, datasets):
    """Test that encoded coordinate data makes it through OpenDAP correctly.

    xref: https://github.com/xpublish-community/xpublish/discussions/246
    """
    dataset_id = "min_coordinates_encoding"
    disk = datasets[dataset_id]

    url = f"{xpublish_server}/datasets/{dataset_id}/opendap"
    served = xr.open_dataset(url)
    xr.testing.assert_identical(served, disk)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NetCDF4 is failing on Windows Github Actions workers",
)
@pytest.mark.parametrize(
    "dataset_id",
    [
        "air",
        "attrs_quote",
        "attrs_cast",
        pytest.param(
            "min_coordinates_encoding",
            marks=pytest.mark.xfail(
                raises=IndexError,
                reason="Unable to decode ocean_time",
                strict=True,
            ),
        ),
        "PRISM_v2_slice",
    ],
)
def test_compare_datasets(xpublish_server, datasets, dataset_id):
    """Compare datasets between OpenDAP and locally opened."""
    disk = datasets[dataset_id]

    url = f"{xpublish_server}/datasets/{dataset_id}/opendap"
    served = xr.open_dataset(url)

    xr.testing.assert_equal(served, disk)
