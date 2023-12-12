"""Test OpenDAP server with air temperature dataset."""
import xarray.tutorial
import xpublish
import numpy as np

from xpublish_opendap import OpenDapPlugin

ds = xarray.tutorial.open_dataset("air_temperature")

ds_attrs_quote = xarray.tutorial.open_dataset("air_temperature")
ds_attrs_quote.attrs["quotes"] = 'This attribute uses "quotes"'
ds_attrs_quote.attrs["npint"] = np.int16(16)
ds_attrs_quote.attrs["npintthirtytwo"] = np.int32(32)

rest = xpublish.Rest(
    {"air": ds, "attrs_quote_types": ds_attrs_quote},
    plugins={"opendap": OpenDapPlugin()},
)

rest.serve()
