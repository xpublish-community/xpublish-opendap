"""Test OpenDAP server with air temperature dataset."""
import xarray.tutorial
import xpublish

from xpublish_opendap import OpenDapPlugin

ds = xarray.tutorial.open_dataset("air_temperature")

rest = xpublish.Rest({"air": ds}, plugins={"opendap": OpenDapPlugin()})

rest.serve()
