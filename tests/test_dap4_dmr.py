# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/dap4/dmr.py — DMR XML generation."""

import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from xpublish_opendap.dap.dap4.dmr import (
    DAP4_NS,
    _attribute_dap4_type,
    generate_dmr,
)
from xpublish_opendap.dap.types import DapType


@pytest.fixture
def basic_ds():
    """Simple dataset with coords and data vars."""
    return xr.Dataset(
        {
            "temp": xr.DataArray(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64"),
                dims=["y", "x"],
                attrs={"units": "kelvin", "long_name": "Temperature"},
            ),
        },
        coords={
            "x": np.array([10.0, 20.0], dtype="float32"),
            "y": np.array([30.0, 40.0], dtype="float32"),
        },
        attrs={"title": "Test dataset", "history": "Created for testing"},
    )


@pytest.fixture
def scalar_ds():
    """Dataset with scalar variables."""
    return xr.Dataset(
        {
            "value": xr.DataArray(np.float64(42.0)),
        },
    )


@pytest.fixture
def multi_type_ds():
    """Dataset with various numpy types."""
    return xr.Dataset(
        {
            "i8": xr.DataArray(np.array([1, 2], dtype="int8"), dims=["x"]),
            "i64": xr.DataArray(np.array([1, 2], dtype="int64"), dims=["x"]),
            "u64": xr.DataArray(np.array([1, 2], dtype="uint64"), dims=["x"]),
            "u16": xr.DataArray(np.array([1, 2], dtype="uint16"), dims=["x"]),
        },
        coords={"x": np.array([0, 1], dtype="int32")},
    )


def _parse_dmr(ds, name="test"):
    """Generate DMR and parse as XML."""
    xml_str = generate_dmr(ds, name)
    return ET.fromstring(xml_str)


class TestDMRStructure:
    def test_root_element(self, basic_ds):
        root = _parse_dmr(basic_ds)
        assert root.tag == f"{{{DAP4_NS}}}Dataset"
        assert root.get("dapVersion") == "4.0"
        assert root.get("dmrVersion") == "1.0"
        assert root.get("name") == "test"

    def test_namespace(self, basic_ds):
        root = _parse_dmr(basic_ds)
        # ET absorbs xmlns into the tag as {ns}tag, so check the tag itself
        assert root.tag == f"{{{DAP4_NS}}}Dataset"

    def test_little_endian_attribute(self, basic_ds):
        root = _parse_dmr(basic_ds)
        attrs = root.findall(f"{{{DAP4_NS}}}Attribute")
        endian_attr = None
        for attr in attrs:
            if attr.get("name") == "_DAP4_Little_Endian":
                endian_attr = attr
                break
        assert endian_attr is not None
        assert endian_attr.get("type") == "UInt8"
        assert endian_attr.find(f"{{{DAP4_NS}}}Value").text == "1"

    def test_valid_xml(self, basic_ds):
        xml_str = generate_dmr(basic_ds, "test")
        assert xml_str.startswith("<?xml version='1.0' encoding=")


class TestDimensions:
    def test_dimension_declarations(self, basic_ds):
        root = _parse_dmr(basic_ds)
        dims = root.findall(f"{{{DAP4_NS}}}Dimension")
        dim_names = {d.get("name") for d in dims}
        assert "x" in dim_names
        assert "y" in dim_names

    def test_dimension_sizes(self, basic_ds):
        root = _parse_dmr(basic_ds)
        dims = root.findall(f"{{{DAP4_NS}}}Dimension")
        sizes = {d.get("name"): int(d.get("size")) for d in dims}
        assert sizes["x"] == 2
        assert sizes["y"] == 2


class TestVariableTypes:
    def test_float32_coord(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float32s = root.findall(f"{{{DAP4_NS}}}Float32")
        names = {el.get("name") for el in float32s}
        assert "x" in names
        assert "y" in names

    def test_float64_data_var(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float64s = root.findall(f"{{{DAP4_NS}}}Float64")
        names = {el.get("name") for el in float64s}
        assert "temp" in names

    def test_int8_type(self, multi_type_ds):
        root = _parse_dmr(multi_type_ds)
        int8s = root.findall(f"{{{DAP4_NS}}}Int8")
        names = {el.get("name") for el in int8s}
        assert "i8" in names

    def test_int64_type(self, multi_type_ds):
        root = _parse_dmr(multi_type_ds)
        int64s = root.findall(f"{{{DAP4_NS}}}Int64")
        names = {el.get("name") for el in int64s}
        assert "i64" in names

    def test_uint64_type(self, multi_type_ds):
        root = _parse_dmr(multi_type_ds)
        uint64s = root.findall(f"{{{DAP4_NS}}}UInt64")
        names = {el.get("name") for el in uint64s}
        assert "u64" in names

    def test_uint16_type(self, multi_type_ds):
        root = _parse_dmr(multi_type_ds)
        uint16s = root.findall(f"{{{DAP4_NS}}}UInt16")
        names = {el.get("name") for el in uint16s}
        assert "u16" in names

    def test_scalar_variable(self, scalar_ds):
        root = _parse_dmr(scalar_ds)
        float64s = root.findall(f"{{{DAP4_NS}}}Float64")
        names = {el.get("name") for el in float64s}
        assert "value" in names
        # Scalar should have no Dim children
        for el in float64s:
            if el.get("name") == "value":
                assert el.findall(f"{{{DAP4_NS}}}Dim") == []


class TestDimReferences:
    def test_coord_has_dim_refs(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float32s = root.findall(f"{{{DAP4_NS}}}Float32")
        for el in float32s:
            if el.get("name") == "x":
                dims = el.findall(f"{{{DAP4_NS}}}Dim")
                assert len(dims) == 1
                assert dims[0].get("name") == "/x"

    def test_data_var_has_dim_refs(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float64s = root.findall(f"{{{DAP4_NS}}}Float64")
        for el in float64s:
            if el.get("name") == "temp":
                dims = el.findall(f"{{{DAP4_NS}}}Dim")
                assert len(dims) == 2
                dim_names = [d.get("name") for d in dims]
                assert dim_names == ["/y", "/x"]


class TestMapElements:
    def test_data_var_has_maps(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float64s = root.findall(f"{{{DAP4_NS}}}Float64")
        for el in float64s:
            if el.get("name") == "temp":
                maps = el.findall(f"{{{DAP4_NS}}}Map")
                assert len(maps) == 2
                map_names = {m.get("name") for m in maps}
                assert "/y" in map_names
                assert "/x" in map_names

    def test_coord_has_no_maps(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float32s = root.findall(f"{{{DAP4_NS}}}Float32")
        for el in float32s:
            maps = el.findall(f"{{{DAP4_NS}}}Map")
            assert len(maps) == 0


class TestAttributes:
    def test_variable_attributes(self, basic_ds):
        root = _parse_dmr(basic_ds)
        float64s = root.findall(f"{{{DAP4_NS}}}Float64")
        for el in float64s:
            if el.get("name") == "temp":
                attrs = el.findall(f"{{{DAP4_NS}}}Attribute")
                attr_names = {a.get("name") for a in attrs}
                assert "units" in attr_names
                assert "long_name" in attr_names

                # Check attribute type and value
                for attr in attrs:
                    if attr.get("name") == "units":
                        assert attr.get("type") == "String"
                        assert attr.find(f"{{{DAP4_NS}}}Value").text == "kelvin"

    def test_global_attributes(self, basic_ds):
        root = _parse_dmr(basic_ds)
        attrs = root.findall(f"{{{DAP4_NS}}}Attribute")
        nc_global = None
        for attr in attrs:
            if attr.get("name") == "NC_GLOBAL":
                nc_global = attr
                break
        assert nc_global is not None
        assert nc_global.get("type") == "Container"

        inner_attrs = nc_global.findall(f"{{{DAP4_NS}}}Attribute")
        inner_names = {a.get("name") for a in inner_attrs}
        assert "title" in inner_names
        assert "history" in inner_names

    def test_no_global_attrs_when_empty(self):
        ds = xr.Dataset({"x": xr.DataArray([1.0])})
        root = _parse_dmr(ds)
        attrs = root.findall(f"{{{DAP4_NS}}}Attribute")
        nc_global_found = any(a.get("name") == "NC_GLOBAL" for a in attrs)
        assert not nc_global_found


class TestMultiTypeAttributes:
    """Tests for DMR attribute type/value formatting helpers."""

    @pytest.fixture
    def attr_ds(self):
        """Dataset with diverse global attribute types."""
        return xr.Dataset(
            {"v": xr.DataArray([1.0], dims=["x"])},
            coords={"x": [0]},
            attrs={
                "array_attr": np.array([1.0, 2.0, 3.0]),
                "bool_attr": True,
                "nan_attr": float("nan"),
                "list_int_attr": [10, 20, 30],
                "list_str_attr": ["alpha", "beta"],
            },
        )

    def _get_nc_global_attr(self, root, attr_name):
        """Find an attribute inside NC_GLOBAL container by name."""
        for container in root.findall(f"{{{DAP4_NS}}}Attribute"):
            if container.get("name") == "NC_GLOBAL":
                for attr in container.findall(f"{{{DAP4_NS}}}Attribute"):
                    if attr.get("name") == attr_name:
                        return attr
        return None

    def test_array_attr_type_and_values(self, attr_ds):
        root = _parse_dmr(attr_ds)
        attr = self._get_nc_global_attr(root, "array_attr")
        assert attr is not None
        assert attr.get("type") == "Float64"
        values = [v.text for v in attr.findall(f"{{{DAP4_NS}}}Value")]
        assert values == ["1.0", "2.0", "3.0"]

    def test_bool_attr_type_and_value(self, attr_ds):
        root = _parse_dmr(attr_ds)
        attr = self._get_nc_global_attr(root, "bool_attr")
        assert attr is not None
        assert attr.get("type") == "UInt8"
        assert attr.find(f"{{{DAP4_NS}}}Value").text == "1"

    def test_list_int_attr_type(self, attr_ds):
        root = _parse_dmr(attr_ds)
        attr = self._get_nc_global_attr(root, "list_int_attr")
        assert attr is not None
        assert attr.get("type") == "Float64"
        values = [v.text for v in attr.findall(f"{{{DAP4_NS}}}Value")]
        assert values == ["10", "20", "30"]

    def test_list_str_attr_type(self, attr_ds):
        root = _parse_dmr(attr_ds)
        attr = self._get_nc_global_attr(root, "list_str_attr")
        assert attr is not None
        assert attr.get("type") == "String"

    def test_numpy_int_attr(self):
        ds = xr.Dataset(
            {"v": xr.DataArray([1.0], dims=["x"])},
            coords={"x": [0]},
            attrs={"np_int": np.int32(42)},
        )
        root = _parse_dmr(ds)
        for container in root.findall(f"{{{DAP4_NS}}}Attribute"):
            if container.get("name") == "NC_GLOBAL":
                for attr in container.findall(f"{{{DAP4_NS}}}Attribute"):
                    if attr.get("name") == "np_int":
                        assert attr.get("type") == "Int32"
                        assert attr.find(f"{{{DAP4_NS}}}Value").text == "42"
                        return
        pytest.fail("np_int attribute not found in DMR")

    def test_numpy_float_attr(self):
        ds = xr.Dataset(
            {"v": xr.DataArray([1.0], dims=["x"])},
            coords={"x": [0]},
            attrs={"np_flt": np.float64(3.14)},
        )
        root = _parse_dmr(ds)
        for container in root.findall(f"{{{DAP4_NS}}}Attribute"):
            if container.get("name") == "NC_GLOBAL":
                for attr in container.findall(f"{{{DAP4_NS}}}Attribute"):
                    if attr.get("name") == "np_flt":
                        assert attr.get("type") == "Float64"
                        assert attr.find(f"{{{DAP4_NS}}}Value").text == "3.14"
                        return
        pytest.fail("np_flt attribute not found in DMR")

    def test_python_int_attr(self):
        ds = xr.Dataset(
            {"v": xr.DataArray([1.0], dims=["x"])},
            coords={"x": [0]},
            attrs={"py_int": 99},
        )
        root = _parse_dmr(ds)
        for container in root.findall(f"{{{DAP4_NS}}}Attribute"):
            if container.get("name") == "NC_GLOBAL":
                for attr in container.findall(f"{{{DAP4_NS}}}Attribute"):
                    if attr.get("name") == "py_int":
                        assert attr.get("type") == "Int64"
                        assert attr.find(f"{{{DAP4_NS}}}Value").text == "99"
                        return
        pytest.fail("py_int attribute not found in DMR")

    def test_complex_ndarray_attr_falls_back_to_string(self):
        ds = xr.Dataset(
            {"v": xr.DataArray([1.0], dims=["x"])},
            coords={"x": [0]},
            attrs={"c": np.array([1 + 2j])},
        )
        root = _parse_dmr(ds)
        for container in root.findall(f"{{{DAP4_NS}}}Attribute"):
            if container.get("name") == "NC_GLOBAL":
                for attr in container.findall(f"{{{DAP4_NS}}}Attribute"):
                    if attr.get("name") == "c":
                        assert attr.get("type") == "String"
                        return
        pytest.fail("c attribute not found in DMR")

    def test_unknown_type_attr_falls_back_to_string(self):
        ds = xr.Dataset(
            {"v": xr.DataArray([1.0], dims=["x"])},
            coords={"x": [0]},
            attrs={"s": frozenset([1, 2])},
        )
        root = _parse_dmr(ds)
        for container in root.findall(f"{{{DAP4_NS}}}Attribute"):
            if container.get("name") == "NC_GLOBAL":
                for attr in container.findall(f"{{{DAP4_NS}}}Attribute"):
                    if attr.get("name") == "s":
                        assert attr.get("type") == "String"
                        return
        pytest.fail("s attribute not found in DMR")

    def test_nan_attr_value(self, attr_ds):
        root = _parse_dmr(attr_ds)
        attr = self._get_nc_global_attr(root, "nan_attr")
        assert attr is not None
        assert attr.get("type") == "Float64"
        assert attr.find(f"{{{DAP4_NS}}}Value").text == "nan"


class TestDap4NameNoneGuards:
    """Tests for defensive guards when DapType.dap4_name is None."""

    _BAD_TYPE = DapType(
        dap2_name="Float64",
        xdr_format=">f8",
        xdr_wire_size=8,
        numpy_dtype=np.dtype("float64"),
        dap4_name=None,
    )

    def test_coord_with_no_dap4_name_raises(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0])})
        with patch(
            "xpublish_opendap.dap.dap4.dmr.resolve_dap_type",
            return_value=self._BAD_TYPE,
        ):
            with pytest.raises(ValueError, match="has no DAP4 name"):
                generate_dmr(ds, "test")

    def test_data_var_with_no_dap4_name_raises(self):
        ds = xr.Dataset({"v": xr.DataArray(np.float64(42.0))})
        with patch(
            "xpublish_opendap.dap.dap4.dmr.resolve_dap_type",
            return_value=self._BAD_TYPE,
        ):
            with pytest.raises(ValueError, match="has no DAP4 name"):
                generate_dmr(ds, "test")

    def test_np_integer_attr_no_dap4_name_returns_string(self):
        with patch(
            "xpublish_opendap.dap.dap4.dmr.resolve_dap_type",
            return_value=self._BAD_TYPE,
        ):
            assert _attribute_dap4_type(np.int32(42)) == "String"

    def test_np_floating_attr_no_dap4_name_returns_string(self):
        with patch(
            "xpublish_opendap.dap.dap4.dmr.resolve_dap_type",
            return_value=self._BAD_TYPE,
        ):
            assert _attribute_dap4_type(np.float32(3.14)) == "String"

    def test_ndarray_attr_no_dap4_name_returns_string(self):
        with patch(
            "xpublish_opendap.dap.dap4.dmr.resolve_dap_type",
            return_value=self._BAD_TYPE,
        ):
            assert _attribute_dap4_type(np.array([1.0, 2.0])) == "String"
