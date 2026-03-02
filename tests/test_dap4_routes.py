# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for DAP4 plugin routes — .dmr, .dap, .dsr."""

import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import xarray as xr
import xpublish
from fastapi.testclient import TestClient

from xpublish_opendap import OpenDapPlugin
from xpublish_opendap.dap.dap4.data import (
    CHUNK_END,
    CHUNK_SIZE_MASK,
    DMR_DATA_SEPARATOR,
)
from xpublish_opendap.dap.dap4.dmr import DAP4_NS
from xpublish_opendap.dap.dap4.headers import CONTENT_TYPES as DAP4_CONTENT_TYPES


@pytest.fixture(scope="module")
def ds():
    """Dataset for route tests."""
    return xr.Dataset(
        {
            "temp": xr.DataArray(
                np.arange(60, dtype="float64").reshape(3, 4, 5),
                dims=["time", "y", "x"],
                attrs={"units": "kelvin"},
            ),
        },
        coords={
            "time": np.arange(3, dtype="float64"),
            "y": np.arange(4, dtype="float32"),
            "x": np.arange(5, dtype="float32"),
        },
        attrs={"title": "Test"},
    )


@pytest.fixture(scope="module")
def client(ds):
    rest = xpublish.Rest({"test": ds}, plugins={"opendap": OpenDapPlugin()})
    return TestClient(rest.app)


class TestDMRRoute:
    def test_dmr_status_200(self, client):
        resp = client.get("/datasets/test/opendap.dmr")
        assert resp.status_code == 200

    def test_dmr_content_type(self, client):
        resp = client.get("/datasets/test/opendap.dmr")
        assert DAP4_CONTENT_TYPES["dmr"] in resp.headers["content-type"]

    def test_dmr_dap4_headers(self, client):
        resp = client.get("/datasets/test/opendap.dmr")
        assert "xpublish-opendap" in resp.headers.get("XOPeNDAP-Server", "")
        assert resp.headers.get("XDAP") == "4.0"

    def test_dmr_valid_xml(self, client):
        resp = client.get("/datasets/test/opendap.dmr")
        root = ET.fromstring(resp.text)
        assert root.tag == f"{{{DAP4_NS}}}Dataset"
        assert root.get("name") == "test"

    def test_dmr_dimensions(self, client):
        resp = client.get("/datasets/test/opendap.dmr")
        root = ET.fromstring(resp.text)
        dims = root.findall(f"{{{DAP4_NS}}}Dimension")
        dim_map = {d.get("name"): int(d.get("size", "0")) for d in dims}
        assert dim_map["time"] == 3
        assert dim_map["y"] == 4
        assert dim_map["x"] == 5

    def test_dmr_variables(self, client):
        resp = client.get("/datasets/test/opendap.dmr")
        root = ET.fromstring(resp.text)
        float64s = root.findall(f"{{{DAP4_NS}}}Float64")
        names = {el.get("name") for el in float64s}
        assert "time" in names
        assert "temp" in names

    def test_dmr_with_constraint(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/temp[0:1][0:1][0:1]")
        assert resp.status_code == 200
        root = ET.fromstring(resp.text)
        dims = root.findall(f"{{{DAP4_NS}}}Dimension")
        dim_map = {d.get("name"): int(d.get("size", "0")) for d in dims}
        assert dim_map["time"] == 2
        assert dim_map["y"] == 2
        assert dim_map["x"] == 2

    def test_dmr_constraint_with_semicolons(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/time;/y")
        assert resp.status_code == 200
        root = ET.fromstring(resp.text)
        # Should have time, y coords, and no data vars
        all_typed_els = []
        for type_name in ("Float64", "Float32", "Int32", "Int64", "UInt8"):
            all_typed_els.extend(root.findall(f"{{{DAP4_NS}}}{type_name}"))
        var_names = {el.get("name") for el in all_typed_els}
        assert "time" in var_names
        assert "y" in var_names
        # temp should not be present (not in projection)
        assert "temp" not in var_names


class TestDAPDataRoute:
    def test_dap_status_200(self, client):
        resp = client.get("/datasets/test/opendap.dap")
        assert resp.status_code == 200

    def test_dap_content_type(self, client):
        resp = client.get("/datasets/test/opendap.dap")
        assert DAP4_CONTENT_TYPES["dap"] in resp.headers["content-type"]

    def test_dap_contains_dmr_and_binary(self, client):
        resp = client.get("/datasets/test/opendap.dap")
        data = resp.content
        assert DMR_DATA_SEPARATOR in data
        idx = data.index(DMR_DATA_SEPARATOR)
        dmr = data[:idx].decode("utf-8")
        assert "<Dataset" in dmr
        binary = data[idx + len(DMR_DATA_SEPARATOR) :]
        assert len(binary) > 0

    def test_dap_ends_with_end_chunk(self, client):
        resp = client.get("/datasets/test/opendap.dap")
        data = resp.content
        idx = data.index(DMR_DATA_SEPARATOR)
        binary = data[idx + len(DMR_DATA_SEPARATOR) :]
        last_4 = binary[-4:]
        raw = struct.unpack(">I", last_4)[0]
        chunk_type = raw & ~CHUNK_SIZE_MASK & ~0x04000000
        assert chunk_type == CHUNK_END

    def test_dap_with_constraint(self, client):
        resp = client.get("/datasets/test/opendap.dap?dap4.ce=/temp[0:0][0:0][0:0]")
        assert resp.status_code == 200
        data = resp.content
        idx = data.index(DMR_DATA_SEPARATOR)
        dmr = data[:idx].decode("utf-8")
        assert "temp" in dmr

    def test_dap_dap4_headers(self, client):
        resp = client.get("/datasets/test/opendap.dap")
        assert resp.headers.get("XDAP") == "4.0"


class TestDSRRoute:
    def test_dsr_status_200(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        assert resp.status_code == 200

    def test_dsr_content_type(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        assert DAP4_CONTENT_TYPES["dsr"] in resp.headers["content-type"]

    def test_dsr_valid_xml(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        root = ET.fromstring(resp.text)
        assert root.tag == f"{{{DAP4_NS}}}DatasetServices"

    def test_dsr_has_services(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        root = ET.fromstring(resp.text)
        services = root.findall(f"{{{DAP4_NS}}}Service")
        assert len(services) >= 2  # At least DMR and DAP

    def test_dsr_service_links(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        root = ET.fromstring(resp.text)
        services = root.findall(f"{{{DAP4_NS}}}Service")
        links = []
        for svc in services:
            link = svc.find(f"{{{DAP4_NS}}}Link")
            if link is not None:
                links.append(link.get("href", ""))
        assert any(".dmr" in href for href in links)
        assert any(".dap" in href for href in links)


class TestDAP4ErrorHandling:
    def test_bad_constraint_returns_dap4_error(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/temp[abc]")
        assert resp.status_code == 400
        root = ET.fromstring(resp.text)
        assert root.tag == f"{{{DAP4_NS}}}Error"
        error_code = root.find(f"{{{DAP4_NS}}}ErrorCode")
        assert error_code is not None

    def test_variable_not_found_returns_dap4_error(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/nonexistent")
        assert resp.status_code == 400
        root = ET.fromstring(resp.text)
        assert root.tag == f"{{{DAP4_NS}}}Error"

    def test_index_out_of_range_returns_dap4_error(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/temp[0:999][0:0][0:0]")
        assert resp.status_code == 400
        root = ET.fromstring(resp.text)
        error_code = root.find(f"{{{DAP4_NS}}}ErrorCode")
        assert error_code is not None
        assert error_code.text == "1003"

    def test_dap_route_memory_limit(self):
        """Plugin with very low memory limit should reject large DAP4 requests."""
        ds_big = xr.Dataset(
            {
                "big": xr.DataArray(
                    np.zeros((100, 100), dtype="float64"), dims=["y", "x"]
                )
            },
            coords={
                "y": np.arange(100, dtype="float64"),
                "x": np.arange(100, dtype="float64"),
            },
        )
        plugin = OpenDapPlugin(max_request_memory_bytes=1)
        rest = xpublish.Rest({"test": ds_big}, plugins={"opendap": plugin})
        c = TestClient(rest.app)

        resp = c.get("/datasets/test/opendap.dap")
        assert resp.status_code == 413
        root = ET.fromstring(resp.text)
        assert root.tag == f"{{{DAP4_NS}}}Error"


class TestDAP4ErrorResponseBody:
    """Validate structure and content of DAP4 error XML responses."""

    def test_error_root_element(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/nonexistent")
        root = ET.fromstring(resp.text)
        assert root.tag == f"{{{DAP4_NS}}}Error"
        assert "httpcode" in root.attrib

    def test_error_code_element(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/nonexistent")
        root = ET.fromstring(resp.text)
        error_code = root.find(f"{{{DAP4_NS}}}ErrorCode")
        assert error_code is not None
        assert error_code.text == "1002"

    def test_error_message_element(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/nonexistent")
        root = ET.fromstring(resp.text)
        message = root.find(f"{{{DAP4_NS}}}Message")
        assert message is not None
        assert message.text is not None
        assert "nonexistent" in message.text

    def test_variable_not_found_error_content(self, client):
        resp = client.get("/datasets/test/opendap.dap?dap4.ce=/missing_var")
        assert resp.status_code == 400
        root = ET.fromstring(resp.text)
        error_code = root.find(f"{{{DAP4_NS}}}ErrorCode")
        assert error_code is not None
        assert error_code.text == "1002"
        message = root.find(f"{{{DAP4_NS}}}Message")
        assert message is not None
        assert message.text is not None
        assert "missing_var" in message.text

    def test_413_error_httpcode(self):
        """Memory limit error should have httpcode='413' and code 1004."""
        ds_big = xr.Dataset(
            {
                "big": xr.DataArray(
                    np.zeros((100, 100), dtype="float64"), dims=["y", "x"]
                )
            },
            coords={
                "y": np.arange(100, dtype="float64"),
                "x": np.arange(100, dtype="float64"),
            },
        )
        plugin = OpenDapPlugin(max_request_memory_bytes=1)
        rest = xpublish.Rest({"test": ds_big}, plugins={"opendap": plugin})
        c = TestClient(rest.app)

        resp = c.get("/datasets/test/opendap.dap")
        assert resp.status_code == 413
        root = ET.fromstring(resp.text)
        assert root.get("httpcode") == "413"
        error_code = root.find(f"{{{DAP4_NS}}}ErrorCode")
        assert error_code is not None
        assert error_code.text == "1004"


class TestHelpIncludesDAP4:
    def test_help_mentions_dap4(self, client):
        resp = client.get("/datasets/test/opendap.help")
        assert ".dmr" in resp.text
        assert ".dap" in resp.text
        assert ".dsr" in resp.text


class TestCrossProtocolConsistency:
    """Verify DAP2 and DAP4 metadata is consistent."""

    def test_variable_names_match_dds_dmr(self, client):
        dds_resp = client.get("/datasets/test/opendap.dds")
        dmr_resp = client.get("/datasets/test/opendap.dmr")

        # Parse DDS variable names (lines containing type declarations)
        dds_text = dds_resp.text
        dds_vars = set()
        for line in dds_text.split("\n"):
            stripped = line.strip()
            for type_prefix in (
                "Float64",
                "Float32",
                "Int32",
                "Int16",
                "UInt16",
                "String",
                "Byte",
            ):
                if stripped.startswith(type_prefix + " "):
                    # Extract var name (before [ or ;)
                    rest = stripped[len(type_prefix) + 1 :]
                    name = rest.split("[")[0].split(";")[0].strip()
                    dds_vars.add(name)

        # Parse DMR variable names
        root = ET.fromstring(dmr_resp.text)
        dmr_vars = set()
        for type_name in (
            "Float64",
            "Float32",
            "Int32",
            "Int16",
            "UInt16",
            "UInt8",
            "Int8",
            "Int64",
            "UInt64",
            "String",
        ):
            for el in root.findall(f"{{{DAP4_NS}}}{type_name}"):
                dmr_vars.add(el.get("name"))

        # Both should have the same variable names
        assert dds_vars == dmr_vars

    def test_dimension_sizes_match(self, client):
        dds_resp = client.get("/datasets/test/opendap.dds")
        dmr_resp = client.get("/datasets/test/opendap.dmr")

        # Parse DMR dimension sizes
        root = ET.fromstring(dmr_resp.text)
        dims = root.findall(f"{{{DAP4_NS}}}Dimension")
        dmr_dim_sizes = {d.get("name"): int(d.get("size", "0")) for d in dims}

        # Parse DDS dimension sizes from coordinate declarations
        # e.g. "Float64 time[time = 3];" → time: 3
        dds_text = dds_resp.text
        dds_dim_sizes = {}
        for match in re.finditer(r"\[(\w+) = (\d+)\]", dds_text):
            dim_name = match.group(1)
            dim_size = int(match.group(2))
            dds_dim_sizes[dim_name] = dim_size

        for dim_name, dmr_size in dmr_dim_sizes.items():
            assert dim_name in dds_dim_sizes, f"Dimension {dim_name} missing from DDS"
            assert dds_dim_sizes[dim_name] == dmr_size


class TestDAP4MultiVar:
    def test_dap4_multi_var_data(self, client):
        resp = client.get("/datasets/test/opendap.dap?dap4.ce=/temp;/time")
        assert resp.status_code == 200
        data = resp.content
        idx = data.index(DMR_DATA_SEPARATOR)
        dmr = data[:idx].decode("utf-8")
        assert "temp" in dmr
        assert "time" in dmr


class TestDAP4Stride:
    def test_dap4_stride_response(self, client):
        resp = client.get(
            "/datasets/test/opendap.dap?dap4.ce=/temp[0:2:2][0:1:3][0:1:4]"
        )
        assert resp.status_code == 200
        data = resp.content
        idx = data.index(DMR_DATA_SEPARATOR)
        dmr = data[:idx].decode("utf-8")
        root = ET.fromstring(dmr)
        # Dimensions should reflect stride subsetting
        dims = root.findall(f"{{{DAP4_NS}}}Dimension")
        dim_map = {d.get("name"): int(d.get("size", "0")) for d in dims}
        # time: [0:2:2] → indices 0, 2 → 2 elements
        assert dim_map["time"] == 2


class TestDSRServices:
    def test_dsr_service_names(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        root = ET.fromstring(resp.text)
        services = root.findall(f"{{{DAP4_NS}}}Service")
        titles = {svc.get("title") for svc in services}
        assert "Dataset Metadata Response" in titles
        assert "DAP4 Data Response" in titles

    def test_dsr_service_base_url(self, client):
        resp = client.get("/datasets/test/opendap.dsr")
        root = ET.fromstring(resp.text)
        services = root.findall(f"{{{DAP4_NS}}}Service")
        links = []
        for svc in services:
            link = svc.find(f"{{{DAP4_NS}}}Link")
            if link is not None:
                links.append(link.get("href", ""))
        assert any(".dmr" in href for href in links)
        assert any(".dap" in href for href in links)
        assert any("opendap" in href for href in links)


class TestDAP2StillWorks:
    """Verify DAP2 routes still function after DAP4 additions."""

    def test_dds_still_works(self, client):
        resp = client.get("/datasets/test/opendap.dds")
        assert resp.status_code == 200
        assert "Dataset {" in resp.text

    def test_das_still_works(self, client):
        resp = client.get("/datasets/test/opendap.das")
        assert resp.status_code == 200
        assert "Attributes {" in resp.text

    def test_dods_still_works(self, client):
        resp = client.get("/datasets/test/opendap.dods")
        assert resp.status_code == 200
        assert b"\nData:\n" in resp.content
