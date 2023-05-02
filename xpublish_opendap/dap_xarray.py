"""Convert xarray.Datasets to OpenDAP datasets."""
import logging
from typing import (
    Any,
)

import numpy as np
import opendap_protocol as dap
import xarray as xr

logger: logging.Logger = logging.getLogger("api")


class NoEncodingDtype(KeyError):
    """Exception for when no encoding dtype is found."""

    ...


dtype_dap = {
    np.ubyte: dap.Byte,
    np.int16: dap.Int16,
    np.uint16: dap.UInt16,
    np.int32: dap.Int32,
    np.uint32: dap.UInt32,
    np.float32: dap.Float32,
    np.float64: dap.Float64,
    np.str_: dap.String,
    np.int64: dap.Float64,  # not a direct mapping
}
dap_dtypes_dict: dict[np.dtype, dap.DAPAtom] = {
    np.dtype(k): v for k, v in dtype_dap.items()
}
del dtype_dap


def dap_dtype(da: xr.DataArray):
    """Return a DAP type for the xr.DataArray."""
    try:
        return dap_dtypes_dict[da.dtype]
    except KeyError as e:
        logger.warning(
            f"Unable to match dtype for {da.name}. "
            f"Going to assume string will work for now... ({e})",
        )
        return dap.String


def dap_attribute(key: str, value: Any) -> dap.Attribute:
    """Create a DAP attribute."""
    if isinstance(value, int):
        dtype = dap.Int32
    elif isinstance(value, float):
        dtype = dap.Float64
    else:
        dtype = dap.String

    return dap.Attribute(
        name=key,
        value=value,
        dtype=dtype,
    )


def dap_dimension(da: xr.DataArray) -> dap.Array:
    """Transform an xarray dimension into a DAP dimension."""
    encoded_da: xr.DataArray = xr.conventions.encode_cf_variable(da.variable)

    dim = dap.Array(
        name=da.name,
        data=encoded_da.values,
        dtype=dap_dtype(encoded_da),
    )

    for key, value in encoded_da.attrs.items():
        dim.append(dap_attribute(key, value))

    return dim


def dap_grid(da: xr.DataArray, dims: dict[str, dap.Array]) -> dap.Grid:
    """Transform an xarray DataArray into a DAP Grid."""
    # get encoding dtype (should exist for numeric dtypes)
    try:
        encoding_dtype = da.encoding["dtype"]
    except KeyError:
        raise NoEncodingDtype

    data_grid = dap.Grid(
        name=da.name,
        data=da.astype(encoding_dtype).data,
        dtype=dap_dtype(da),
        dimensions=[dims[dim] for dim in da.dims],
    )

    for key, value in da.attrs.items():
        data_grid.append(dap_attribute(key, value))

    return data_grid


def dap_dataset(ds: xr.Dataset, name: str) -> dap.Dataset:
    """Create a DAP Dataset for an xarray Dataset."""
    dataset = dap.Dataset(name=name)

    dims: dict[str, dap.Array] = {}
    for dim in ds.dims:
        dims[dim] = dap_dimension(ds[dim])

    dataset.append(*dims.values())

    for var in ds.data_vars:
        try:
            data_grid = dap_grid(ds[var], dims)
            dataset.append(data_grid)
        except NoEncodingDtype:
            logging.warning(
                f"Unable to create grid for {var} of type={ds[var].dtype}. "
                f"No encoding dtype found. "
                f"Skipping this variable for now...",
            )
            continue

    for key, value in ds.attrs.items():
        dataset.append(dap_attribute(key, value))

    return dataset
