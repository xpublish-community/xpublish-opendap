"""Convert xarray.Datasets to OpenDAP datasets."""

import logging
from typing import (
    Any,
    Union,
)

import numpy as np
import opendap_protocol as dap
import xarray as xr

logger: logging.Logger = logging.getLogger("api")

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
dap_dtypes_dict: dict[np.dtype, type[dap.DAPAtom]] = {
    np.dtype(k): v for k, v in dtype_dap.items()
}
del dtype_dap


def dap_dtype(da: Union[xr.DataArray, xr.Variable]) -> type[dap.DAPAtom]:
    """Return a DAP type for the xr.DataArray."""
    try:
        return dap_dtypes_dict[da.dtype]
    except KeyError as e:
        logger.warning(
            f"Unable to match dtype={da.dtype} for {getattr(da, 'name', type(da))}. "
            f"Going to assume string will work for now... ({e})",
        )
        return dap.String


def dap_attribute(key: str, value: Any) -> dap.Attribute:
    """Create a DAP attribute."""
    if isinstance(value, int):
        dtype = dap.Int32
    elif isinstance(value, float):
        dtype = dap.Float64
    elif isinstance(value, np.float32):
        dtype = dap.Float32
    elif isinstance(value, np.int16):
        dtype = dap.Int16
    elif isinstance(value, np.int32):
        dtype = dap.Int32
    elif isinstance(value, str):
        dtype = dap.String
        # Escape a double quote in the attribute value.
        # Other servers like TDS do this. Without this clients fail.
        value = value.replace('"', '\\"')
    else:
        dtype = dap.String

    return dap.Attribute(
        name=key,
        value=value,
        dtype=dtype,
    )


def encode_da(da: xr.DataArray) -> xr.Variable:
    """Encode an Xarray DataArray."""
    if np.issubdtype(da.dtype, np.datetime64):
        encoded_da: xr.Variable = xr.conventions.encode_cf_variable(da.variable)

    else:
        encoded_da = da.variable

    # protect against reverse encoding not matching to dap.DAPAtom dtypes
    if str(encoded_da.dtype).startswith(">"):
        encoded_da = encoded_da.astype(str(encoded_da.dtype).replace(">", "<"))

    return encoded_da


def dap_dimension(da: xr.DataArray) -> dap.Array:
    """Transform an xarray dimension into a DAP dimension."""
    encoded_da = encode_da(da)

    dim = dap.Array(
        name=da.name,
        data=encoded_da.data,
        dtype=dap_dtype(encoded_da),
    )

    for key, value in encoded_da.attrs.items():
        dim.append(dap_attribute(key, value))

    return dim


class Array(dap.DAPDataObject):
    """Generate a OpenDAP array that supports multiple dimensions."""

    def dds(self, constraint="", slicing=None):
        """Generate DDS responses with multiple dimensions."""
        if dap.meets_constraint(constraint, self.data_path):
            # Check for slice
            if slicing is None:
                slices = dap.parse_slice_constraint(constraint)
            else:
                slices = slicing

            yield self.indent + f"{self.dtype()} {self.name}"

            # Yield dimensions
            for i, dim in enumerate(self.dimensions):
                sl = slices[i] if i < len(slices) else ...
                dimlen = int(np.prod(dim.data[sl].shape))
                yield f"[{dim.name} = {dimlen}]"

            yield ";\n"


def dap_grid(da: xr.DataArray, dims: dict[str, dap.Array]) -> dap.Grid:
    """Transform an xarray DataArray into a DAP Grid."""
    encoded_da = encode_da(da)

    dimensions = []
    use_array = False
    for dim in da.dims:
        try:
            dimensions.append(dims[dim])
        except KeyError:
            use_array = True
            dimensions.append(dap_dimension(da[dim]))

    if len(dimensions) == 0:
        use_array = True

    dap_kwargs = {
        "name": da.name,
        "data": encoded_da.astype(encoded_da.dtype).data,
        "dtype": dap_dtype(encoded_da),
        "dimensions": dimensions,
    }

    if use_array:
        data_grid = Array(**dap_kwargs)
    else:
        data_grid = dap.Grid(**dap_kwargs)

    data_grid.append(*dimensions)

    for key, value in da.attrs.items():
        data_grid.append(dap_attribute(key, value))

    try:
        data_grid.append(dap_attribute("coordinates", da.encoding["coordinates"]))
    except KeyError:
        pass
    return data_grid


def dap_dataset(ds: xr.Dataset, name: str) -> dap.Dataset:
    """Create a DAP Dataset for an xarray Dataset."""
    dataset = dap.Dataset(name=name)

    dims: dict[str, dap.Array] = {}
    for dim in ds.coords:
        dims[dim] = dap_dimension(ds[dim])

    dataset.append(*dims.values())

    for var in ds.data_vars:
        data_grid = dap_grid(ds[var], dims)
        dataset.append(data_grid)

    for key, value in ds.attrs.items():
        if key == "_xpublish_id":
            continue
        dataset.append(dap_attribute(key, value))

    return dataset
