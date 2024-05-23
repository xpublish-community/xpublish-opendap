"""Test OpenDAP server with air temperature dataset."""

from pathlib import Path

import numpy as np
import xarray.tutorial
import xpublish

from xpublish_opendap import OpenDapPlugin

ds = xarray.tutorial.open_dataset("air_temperature")

ds_attrs_quote = xarray.tutorial.open_dataset("air_temperature")
ds_attrs_quote.attrs["quotes"] = 'This attribute uses "quotes"'
ds_attrs_cast = xarray.tutorial.open_dataset("air_temperature")
ds_attrs_cast.attrs["npint"] = np.int16(16)
ds_attrs_cast.attrs["npintthirtytwo"] = np.int32(32)

ds_coordinates_encoding = xarray.open_dataset(
    Path(__file__).parent / "min_coordinates_encoding.nc",
)

rest = xpublish.Rest(
    {
        "air": ds,
        "attrs_quote": ds_attrs_quote,
        "attrs_cast": ds_attrs_cast,
        "coords_encoding": ds_coordinates_encoding,
    },
    plugins={"opendap": OpenDapPlugin()},
)

rest.serve()
