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
