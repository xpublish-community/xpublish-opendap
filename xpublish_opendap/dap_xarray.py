"""
Convert xarray.Datasets to OpenDAP datasets
"""
import logging

import numpy as np
import opendap_protocol as dap
import xarray as xr

logger = logging.getLogger("api")

dtype_dap = {
    np.ubyte: dap.Byte,
    np.int16: dap.Int16,
    np.uint16: dap.UInt16,
    np.int32: dap.Int32,
    np.uint32: dap.UInt32,
    np.float32: dap.Float32,
    np.float64: dap.Float64,
    np.str_: dap.String,
    # Not a direct mapping
    np.int64: dap.Float64,
}
dtype_dap = {np.dtype(k): v for k, v in dtype_dap.items()}


def dap_dtype(da: xr.DataArray):
    """Return a DAP type for the xr.DataArray"""
    try:
        return dtype_dap[da.dtype]
    except KeyError as e:
        logger.warning(
            f"Unable to match dtype for {da.name}. Going to assume string will work for now... ({e})",
        )
        return dap.String


def dap_dimension(da: xr.DataArray) -> dap.Array:
    """Transform an xarray dimension into a DAP dimension"""
    encoded_da = xr.conventions.encode_cf_variable(da)
    dim = dap.Array(name=da.name, data=encoded_da.values, dtype=dap_dtype(encoded_da))

    for k, v in encoded_da.attrs.items():
        dim.append(dap.Attribute(name=k, value=v, dtype=dap.String))

    return dim


def dap_grid(da: xr.DataArray, dims: dict[str, dap.Array]) -> dap.Grid:
    """Transform an xarray DataArray into a DAP Grid"""
    data_array = dap.Grid(
        name=da.name,
        data=da.astype(da.encoding["dtype"]).data,
        dtype=dap_dtype(da),
        dimensions=[dims[dim] for dim in da.dims],
    )

    for k, v in da.attrs.items():
        data_array.append(dap.Attribute(name=k, value=v, dtype=dap.String))

    return data_array


def dap_dataset(ds: xr.Dataset, name: str) -> dap.Dataset:
    """Create a DAP Dataset for an xarray Dataset"""
    dataset = dap.Dataset(name=name)

    dims = {}
    for dim in ds.dims:
        dims[dim] = dap_dimension(ds[dim])

    dataset.append(*dims.values())

    for var in ds.variables:
        if var not in ds.dims:
            data_array = dap_grid(ds[var], dims)
            dataset.append(data_array)

    for k, v in ds.attrs.items():
        dataset.append(dap.Attribute(name=k, value=v, dtype=dap.String))

    return dataset
