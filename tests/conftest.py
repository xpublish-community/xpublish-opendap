"""Py.test configuration and shared fixtures."""

from pathlib import Path

import pytest
from xarray.tutorial import open_dataset
from xprocess import ProcessStarter

server_path = Path(__file__).parent / "server.py"


@pytest.fixture
def xpublish_server(xprocess):
    """Launch an Xpublish server in the background.

    Server has the air_temperature tutorial dataset
    at `air` and has the OpenDAP plugin running with
    defaults.
    """

    class Starter(ProcessStarter):
        # Wait till the pattern is printed before
        # considering things started
        pattern = "Uvicorn running on"

        # server startup args
        args = ["python", str(server_path)]

        # seconds before timing out on server startup
        timeout = 30

        # Try to cleanup if inturrupted
        terminate_on_interrupt = True

    xprocess.ensure("xpublish", Starter)
    yield "http://0.0.0.0:9000"
    xprocess.getinfo("xpublish").terminate()


@pytest.fixture(scope="session")
def dataset():
    """Xarray air temperature tutorial dataset."""
    ds = open_dataset("air_temperature")

    return ds
