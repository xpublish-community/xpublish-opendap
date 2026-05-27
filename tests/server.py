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

# Dataset with datetime64/timedelta64 *data variables* (not dimension
# coordinates), e.g. CF time-bounds style variables. These have no DAP dtype
# and must be CF-encoded to numeric before serialization. See test_time_vars.
ds_time_vars = xarray.Dataset(
    {
        "temp": (("member", "lead"), np.zeros((3, 4), dtype="float32")),
        "start_time": (
            ("member", "lead"),
            np.broadcast_to(np.datetime64("1965-01-01"), (3, 4)).copy(),
        ),
        "duration": (
            ("member", "lead"),
            np.full((3, 4), np.timedelta64(365, "D")),
        ),
    },
    coords={"member": np.arange(3), "lead": np.arange(4)},
)

rest = xpublish.Rest(
    {
        "air": ds,
        "attrs_quote": ds_attrs_quote,
        "attrs_cast": ds_attrs_cast,
        "time_vars": ds_time_vars,
    },
    plugins={"opendap": OpenDapPlugin()},
)

rest.serve()
