"""Integration test fixtures and helpers.

Uses the xpublish_server fixture from the parent conftest (auto-discovered by pytest)
to run real DAP client tests against a live xpublish server.
"""

import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_url(xpublish_server):
    """Base URL of the running xpublish server."""
    return xpublish_server


@pytest.fixture
def opendap_base(base_url):
    """OpenDAP endpoint for the ``air`` dataset."""
    return f"{base_url}/datasets/air/opendap"


@pytest.fixture(scope="session")
def reference_ds():
    """Reference air_temperature dataset for value comparison."""
    return xr.tutorial.open_dataset("air_temperature")


@pytest.fixture(scope="session")
def reference_nc(tmp_path_factory):
    """Write the air_temperature dataset to a local netCDF4 file once per session.

    Returns the file path as a string (for subprocess compatibility).
    """
    ds = xr.tutorial.open_dataset("air_temperature")
    path = tmp_path_factory.mktemp("reference") / "air_temperature.nc"
    ds.to_netcdf(path)
    ds.close()
    return str(path)
