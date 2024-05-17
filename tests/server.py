"""Test OpenDAP server with air temperature dataset."""

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

rest = xpublish.Rest(
    {"air": ds, "attrs_quote": ds_attrs_quote, "attrs_cast": ds_attrs_cast},
    plugins={"opendap": OpenDapPlugin()},
)

rest.serve()
