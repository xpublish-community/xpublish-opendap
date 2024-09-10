"""Datasets for testing."""

from pathlib import Path

import numpy as np
import xarray as xr

datasets = {}

datasets_path = Path(__file__).parent / "data"

datasets["air"] = xr.open_dataset(datasets_path / "tutorial_air_temperature.nc")

ds_attrs_quote = xr.tutorial.open_dataset("air_temperature")
ds_attrs_quote.attrs["quotes"] = 'This attribute uses "quotes"'

datasets["attrs_quote"] = ds_attrs_quote

ds_attrs_cast = xr.tutorial.open_dataset("air_temperature")
ds_attrs_cast.attrs["npint"] = np.int16(16)
ds_attrs_cast.attrs["npintthirtytwo"] = np.int32(32)

datasets["attrs_cast"] = ds_attrs_cast

datasets["min_coordinates_encoding"] = xr.open_dataset(
    datasets_path / "min_coordinates_encoding.nc",
)

# xref: https://github.com/xpublish-community/xpublish-opendap/issues/59
datasets["PRISM_v2_slice"] = xr.open_dataset(
    datasets_path / "PRISM_v2_slice.nc",
)
