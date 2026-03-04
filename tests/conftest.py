"""Py.test configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
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

    class Starter(ProcessStarter):  # type: ignore[misc]
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


@pytest.fixture(scope="session")
def multi_type_ds():
    """Dataset with diverse dtypes and edge-case attributes."""
    return xr.Dataset(
        {
            "i8_var": xr.DataArray(
                np.array([[-1, 0], [1, 2]], dtype="int8"),
                dims=["y", "x"],
            ),
            "i16_var": xr.DataArray(
                np.array([[100, -200], [300, -400]], dtype="int16"),
                dims=["y", "x"],
            ),
            "u16_var": xr.DataArray(
                np.array([[1000, 2000], [3000, 4000]], dtype="uint16"),
                dims=["y", "x"],
            ),
            "i32_var": xr.DataArray(
                np.array([[100000, -100000], [200000, -200000]], dtype="int32"),
                dims=["y", "x"],
            ),
            "u32_var": xr.DataArray(
                np.array([[3_000_000_000, 0], [1, 4_000_000_000]], dtype="uint32"),
                dims=["y", "x"],
            ),
            "i64_var": xr.DataArray(
                np.array([[1_000_000_000_000, -1], [0, 42]], dtype="int64"),
                dims=["y", "x"],
            ),
            "str_var": xr.DataArray(
                np.array([["hello", "world"], ["foo", "bar"]], dtype=object),
                dims=["y", "x"],
            ),
            "bool_var": xr.DataArray(
                np.array([[True, False], [False, True]]),
                dims=["y", "x"],
            ),
        },
        coords={
            "x": np.array([10, 20], dtype="int32"),
            "y": np.array([30, 40], dtype="int32"),
            "time_coord": xr.DataArray(
                pd.date_range("2000-01-01", periods=2, freq="D"),
                dims=["y"],
            ),
        },
        attrs={
            "title": "Multi-type test",
            "array_attr": np.array([1.0, 2.0, 3.0]),
            "bool_attr": True,
            "nan_attr": float("nan"),
            "inf_attr": float("inf"),
            "empty_str_attr": "",
            "list_int_attr": [10, 20, 30],
            "list_str_attr": ["alpha", "beta"],
        },
    )
