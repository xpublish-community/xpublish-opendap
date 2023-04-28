# ruff: noqa: D100,D103
import pytest
import xpublish
from fastapi.testclient import TestClient

from xpublish_opendap import OpenDapPlugin


@pytest.fixture(scope="session")
def dataset():
    from xarray.tutorial import open_dataset

    ds = open_dataset("air_temperature")

    return ds


@pytest.fixture(scope="session")
def dap_xpublish(dataset):
    rest = xpublish.Rest({"air": dataset}, plugins={"opendap": OpenDapPlugin()})

    return rest


@pytest.fixture(scope="session")
def dap_client(dap_xpublish):
    app = dap_xpublish.app
    client = TestClient(app)

    return client


def test_dds_response(dap_client):
    response = dap_client.get("/datasets/air/opendap.dds")

    assert response.status_code == 200, "Response did not return successfully"

    content = response.content.decode("utf-8")

    assert "Dataset" in content
    assert "Float32 lat[lat = 25]" in content
    assert "Float32 time[time = 2920]" in content
    assert "Float32 lon[lon = 53]" in content
    assert "Grid {" in content
    assert "Float32 air[time = 2920][lat = 25][lon = 53]" in content


def test_das_response(dap_client):
    response = dap_client.get("/datasets/air/opendap.das")

    assert response.status_code == 200, "Response did not return successfully"

    content = response.content.decode("utf-8")

    assert "Attributes" in content
    assert 'String standard_name "latitude"' in content
    assert 'String standard_name "time"' in content
    assert 'String standard_name "longitude"' in content
    assert (
        'String long_name "4xDaily Air temperature at sigma level 995"' in content
    ), "long_name attribute missing from air DataArray"
    assert (
        'String title "4x daily NMC reanalysis (1948)"' in content
    ), "Global attributes are returned"
