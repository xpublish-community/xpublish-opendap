"""DDS (Dataset Descriptor Structure) text generation from xr.Dataset."""

from __future__ import annotations

from collections.abc import Hashable, Iterator

import xarray as xr

from xpublish_opendap.dap.types import cf_encode_variable, resolve_dap_type


def generate_dds(
    ds: xr.Dataset,
    dataset_name: str,
) -> Iterator[str]:
    """Yield DDS text lines for the dataset.

    Coordinates are emitted as top-level Array declarations.
    Data variables with named dimensions are emitted as Grid declarations.
    Scalar variables (0-d) are emitted as bare atomic type declarations.

    Args:
        ds: The (possibly subsetted) xarray Dataset.
        dataset_name: The name to use for the Dataset declaration.

    Yields:
        Lines of DDS text.
    """
    yield "Dataset {\n"

    # Emit ALL coordinates as top-level arrays (standard DAP2 behavior).
    # Coordinates also appear as Maps inside Grid declarations below.
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        encoded = cf_encode_variable(coord.variable)
        dap_type = resolve_dap_type(encoded.dtype)

        if coord.ndim == 0:
            yield f"    {dap_type.dap2_name} {_escape_name(coord_name)};\n"
        else:
            dims_str = "".join(
                f"[{_escape_name(dim)} = {ds.sizes[dim]}]" for dim in coord.dims
            )
            yield f"    {dap_type.dap2_name} {_escape_name(coord_name)}{dims_str};\n"

    # Emit data variables as Grids (if they have dimensions) or scalars
    for var_name in ds.data_vars:
        var = ds[var_name]
        encoded = cf_encode_variable(var.variable)
        dap_type = resolve_dap_type(encoded.dtype)
        escaped_name = _escape_name(var_name)

        if var.ndim == 0:
            # Scalar variable
            yield f"    {dap_type.dap2_name} {escaped_name};\n"
        else:
            # Grid declaration
            dims_str = "".join(
                f"[{_escape_name(dim)} = {ds.sizes[dim]}]" for dim in var.dims
            )

            yield "    Grid {\n"
            yield "      Array:\n"
            yield f"        {dap_type.dap2_name} {escaped_name}{dims_str};\n"
            yield "      Maps:\n"
            for dim in var.dims:
                if dim in ds.coords:
                    dim_coord = ds.coords[dim]
                    dim_encoded = cf_encode_variable(dim_coord.variable)
                    dim_dap_type = resolve_dap_type(dim_encoded.dtype)
                    dim_escaped = _escape_name(dim)
                    yield (
                        f"        {dim_dap_type.dap2_name} "
                        f"{dim_escaped}[{dim_escaped} = {ds.sizes[dim]}];\n"
                    )
            yield f"    }} {escaped_name};\n"

    yield f"}} {_escape_name(dataset_name)};\n"


def _escape_name(name: Hashable) -> str:
    """Escape a variable/dimension name for DAP2 DDS output.

    DAP2 allows alphanumeric, underscore, and a few others.
    Characters that need escaping get percent-encoded.
    """
    # For now, pass through as-is. Most scientific dataset names are safe.
    # A full implementation would percent-encode special chars.
    return str(name)
