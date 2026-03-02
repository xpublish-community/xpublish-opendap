"""DMR (Dataset Metadata Response) XML generation from xr.Dataset.

The DMR is the DAP4 equivalent of the combined DDS+DAS from DAP2.
It describes the dataset structure, types, dimensions, and attributes
in a single XML document.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import xarray as xr

from xpublish_opendap.dap.types import cf_encode_variable, resolve_dap_type

DAP4_NS = 'http://xml.opendap.org/ns/DAP/4.0#'


def generate_dmr(ds: xr.Dataset, dataset_name: str) -> str:
    """Generate a DAP4 DMR XML document for the dataset.

    Structure:
        <Dataset xmlns="..." dapVersion="4.0" name="...">
          <Dimension name="x" size="100"/>
          ...
          <Float64 name="coord">
            <Dim name="/x"/>
            <Attribute name="units" type="String"><Value>degrees</Value></Attribute>
          </Float64>
          ...
          <Float32 name="temp">
            <Dim name="/x"/>
            <Map name="/coord"/>
            <Attribute .../>
          </Float32>
          ...
          <Attribute name="NC_GLOBAL" type="Container">
            <Attribute name="title" type="String"><Value>...</Value></Attribute>
          </Attribute>
        </Dataset>

    Args:
        ds: The (possibly subsetted) xarray Dataset.
        dataset_name: The name to use for the Dataset element.

    Returns:
        XML string of the DMR document.
    """
    root = ET.Element('Dataset')
    root.set('xmlns', DAP4_NS)
    root.set('dapVersion', '4.0')
    root.set('dmrVersion', '1.0')
    root.set('name', dataset_name)

    # Byte order attribute
    _add_attribute(root, '_DAP4_Little_Endian', 'UInt8', '1')

    # Dimension declarations
    for dim_name, dim_size in ds.sizes.items():
        dim_el = ET.SubElement(root, 'Dimension')
        dim_el.set('name', str(dim_name))
        dim_el.set('size', str(dim_size))

    # Coordinate variables
    coord_names = set(ds.coords)
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        encoded = cf_encode_variable(coord.variable)
        dap_type = resolve_dap_type(encoded.dtype, protocol='dap4')

        if dap_type.dap4_name is None:
            raise ValueError(f"DapType for {encoded.dtype!r} has no DAP4 name")
        var_el = ET.SubElement(root, dap_type.dap4_name)
        var_el.set('name', str(coord_name))

        # Dimension references
        for dim in coord.dims:
            dim_ref = ET.SubElement(var_el, 'Dim')
            dim_ref.set('name', f'/{dim}')

        # Variable attributes
        _add_variable_attributes(var_el, encoded.attrs)

    # Data variables
    for var_name in ds.data_vars:
        var = ds[var_name]
        encoded = cf_encode_variable(var.variable)
        dap_type = resolve_dap_type(encoded.dtype, protocol='dap4')

        if dap_type.dap4_name is None:
            raise ValueError(f"DapType for {encoded.dtype!r} has no DAP4 name")
        var_el = ET.SubElement(root, dap_type.dap4_name)
        var_el.set('name', str(var_name))

        # Dimension references
        for dim in var.dims:
            dim_ref = ET.SubElement(var_el, 'Dim')
            dim_ref.set('name', f'/{dim}')

        # Map references to coordinates
        for dim in var.dims:
            if dim in coord_names:
                map_el = ET.SubElement(var_el, 'Map')
                map_el.set('name', f'/{dim}')

        # Variable attributes
        _add_variable_attributes(var_el, encoded.attrs)

    # Global attributes in NC_GLOBAL container
    if ds.attrs:
        nc_global = ET.SubElement(root, 'Attribute')
        nc_global.set('name', 'NC_GLOBAL')
        nc_global.set('type', 'Container')
        for key, value in ds.attrs.items():
            dap4_type = _attribute_dap4_type(value)
            _add_attribute(nc_global, key, dap4_type, _format_attr_value(value))

    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode', xml_declaration=True) + '\n'


def _add_variable_attributes(parent: ET.Element, attrs: dict[str, Any]) -> None:
    """Add attribute elements to a variable element."""
    for key, value in attrs.items():
        dap4_type = _attribute_dap4_type(value)
        _add_attribute(parent, key, dap4_type, _format_attr_value(value))


def _add_attribute(parent: ET.Element, name: str, dap4_type: str, value: str) -> None:
    """Add a single Attribute element with a Value child."""
    attr_el = ET.SubElement(parent, 'Attribute')
    attr_el.set('name', name)
    attr_el.set('type', dap4_type)

    # Handle array-like values (multiple Value elements)
    if '\n' in value and dap4_type != 'String':
        for v in value.split('\n'):
            val_el = ET.SubElement(attr_el, 'Value')
            val_el.text = v
    else:
        val_el = ET.SubElement(attr_el, 'Value')
        val_el.text = value


def _attribute_dap4_type(value: Any) -> str:  # noqa: PLR0911, PLR0912
    """Determine the DAP4 type name for an attribute value."""
    if isinstance(value, str):
        return 'String'
    if isinstance(value, bool):
        return 'UInt8'
    if isinstance(value, np.integer):
        dap_type = resolve_dap_type(np.dtype(type(value)), protocol='dap4')
        if dap_type.dap4_name is None:
            return 'String'
        return dap_type.dap4_name
    if isinstance(value, int):
        return 'Int64'
    if isinstance(value, np.floating):
        dap_type = resolve_dap_type(np.dtype(type(value)), protocol='dap4')
        if dap_type.dap4_name is None:
            return 'String'
        return dap_type.dap4_name
    if isinstance(value, float):
        return 'Float64'
    if isinstance(value, np.ndarray):
        try:
            dap_type = resolve_dap_type(value.dtype, protocol='dap4')
            if dap_type.dap4_name is None:
                raise ValueError(f"DapType for {value.dtype!r} has no DAP4 name")
            return dap_type.dap4_name
        except (ValueError, KeyError):
            return 'String'
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], str):
            return 'String'
        return 'Float64'
    return 'String'


def _format_attr_value(value: Any) -> str:
    """Format an attribute value as text for a DMR Value element."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, (np.ndarray, list, tuple)):
        return '\n'.join(str(v) for v in value)
    return str(value)
