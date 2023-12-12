"""Test OpenDAP server with air temperature dataset."""
import xarray.tutorial
import xpublish

from xpublish_opendap import OpenDapPlugin

ds = xarray.tutorial.open_dataset("air_temperature")

ds_attrs_quote = xarray.tutorial.open_dataset("air_temperature")
ds_attrs_quote.attrs["quotes"] = 'This attribute uses "quotes"'

rest = xpublish.Rest(
    {"air": ds, "attrs_quote": ds_attrs_quote},
    plugins={"opendap": OpenDapPlugin()},
)

rest.serve()
