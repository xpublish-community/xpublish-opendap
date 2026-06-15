"""DAS (Dataset Attribute Structure) text generation from xr.Dataset."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import xarray as xr

from xpublish_opendap.dap.types import cf_encode_variable, resolve_dap_type


def generate_das(ds: xr.Dataset) -> Iterator[str]:
    """Yield DAS text lines for the dataset.

    Each variable (coordinates and data variables) gets its own attribute
    container. Global attributes are placed in an NC_GLOBAL container.

    Args:
        ds: The (possibly subsetted) xarray Dataset.

    Yields:
        Lines of DAS text.
    """
    yield "Attributes {\n"

    # Coordinate variable attributes
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        encoded = cf_encode_variable(coord.variable)
        attrs = encoded.attrs
        if attrs:
            yield f"    {coord_name} {{\n"
            yield from _format_attributes(attrs, indent=8)
            yield "    }\n"

    # Data variable attributes
    for var_name in ds.data_vars:
        var = ds[var_name]
        encoded = cf_encode_variable(var.variable)
        attrs = encoded.attrs
        if attrs:
            yield f"    {var_name} {{\n"
            yield from _format_attributes(attrs, indent=8)
            yield "    }\n"

    # Global attributes in NC_GLOBAL container
    if ds.attrs:
        yield "    NC_GLOBAL {\n"
        yield from _format_attributes(ds.attrs, indent=8)
        yield "    }\n"

    yield "}\n"


def _format_attributes(attrs: dict[str, Any], indent: int = 8) -> Iterator[str]:
    """Format a dict of attributes as DAS attribute lines.

    Args:
        attrs: The attribute dictionary.
        indent: Number of spaces for indentation.

    Yields:
        Formatted attribute lines.
    """
    pad = " " * indent
    for key, value in attrs.items():
        dap_type_name = _attribute_type_name(value)
        formatted_value = _format_attribute_value(value, dap_type_name)
        yield f"{pad}{dap_type_name} {key} {formatted_value};\n"


def _attribute_type_name(value: Any) -> str:  # noqa: PLR0911
    """Determine the DAP2 type name for an attribute value."""
    if isinstance(value, str):
        return "String"
    if isinstance(value, bool):
        return "Byte"
    if isinstance(value, np.integer):
        return resolve_dap_type(np.dtype(type(value))).dap2_name
    if isinstance(value, int):
        return "Int32"
    if isinstance(value, np.floating):
        return resolve_dap_type(np.dtype(type(value))).dap2_name
    if isinstance(value, float):
        return "Float64"
    if isinstance(value, np.ndarray):
        try:
            dap_type = resolve_dap_type(value.dtype)
            return dap_type.dap2_name
        except (ValueError, KeyError):
            return "String"
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], str):
            return "String"
        return "Float64"
    return "String"


def _format_attribute_value(value: Any, dap_type_name: str) -> str:
    """Format an attribute value for DAS output."""
    if dap_type_name == "String":
        return _format_string_value(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _format_string_value(value: Any) -> str:
    """Format a string attribute value with proper quoting."""
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = []
        for v in value:
            escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{escaped}"')
        return ", ".join(parts)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
