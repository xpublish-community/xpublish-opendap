# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for plugin.py — route handlers, headers, errors, constraints."""

import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import xpublish
from fastapi.testclient import TestClient

from xpublish_opendap import OpenDapPlugin
from xpublish_opendap.dap.dap4.dmr import DAP4_NS


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


class TestDDSRoute:
    def test_dds_status_200(self, client):
        resp = client.get("/datasets/test/opendap.dds")
        assert resp.status_code == 200

    def test_dds_content_type(self, client):
        resp = client.get("/datasets/test/opendap.dds")
        assert "text/plain" in resp.headers["content-type"]

    def test_dds_dap_headers(self, client):
        resp = client.get("/datasets/test/opendap.dds")
        assert resp.headers.get("Content-Description") == "dods-dds"
        assert "xpublish-opendap" in resp.headers.get("XDODS-Server", "")

    def test_dds_content(self, client):
        resp = client.get("/datasets/test/opendap.dds")
        text = resp.text
        assert "Dataset {" in text
        assert "Float64 temp[time = 3][y = 4][x = 5]" in text
        assert "Grid {" in text

    def test_dds_with_constraint(self, client):
        resp = client.get("/datasets/test/opendap.dds?temp[0:1][0:1][0:1]")
        text = resp.text
        assert "temp[time = 2][y = 2][x = 2]" in text


class TestDASRoute:
    def test_das_status_200(self, client):
        resp = client.get("/datasets/test/opendap.das")
        assert resp.status_code == 200

    def test_das_content(self, client):
        resp = client.get("/datasets/test/opendap.das")
        text = resp.text
        assert "Attributes {" in text
        assert 'String units "kelvin"' in text
        assert "NC_GLOBAL {" in text
        assert 'String title "Test"' in text

    def test_das_dap_headers(self, client):
        resp = client.get("/datasets/test/opendap.das")
        assert resp.headers.get("Content-Description") == "dods-das"


class TestDODSRoute:
    def test_dods_status_200(self, client):
        resp = client.get("/datasets/test/opendap.dods")
        assert resp.status_code == 200

    def test_dods_content_type(self, client):
        resp = client.get("/datasets/test/opendap.dods")
        assert resp.headers["content-type"] == "application/octet-stream"

    def test_dods_dap_headers(self, client):
        resp = client.get("/datasets/test/opendap.dods")
        assert resp.headers.get("Content-Description") == "dods-data"

    def test_dods_has_dds_and_data(self, client):
        resp = client.get("/datasets/test/opendap.dods")
        assert b"\nData:\n" in resp.content
        parts = resp.content.split(b"\nData:\n")
        dds_text = parts[0].decode("utf-8")
        assert "Dataset {" in dds_text
        assert len(parts[1]) > 0  # binary data present

    def test_dods_with_constraint(self, client):
        resp = client.get("/datasets/test/opendap.dods?temp[0:0][0:0][0:0]")
        assert resp.status_code == 200
        dds_text = resp.content.split(b"\nData:\n")[0].decode("utf-8")
        assert "temp[time = 1][y = 1][x = 1]" in dds_text


class TestVersionRoute:
    def test_version_status_200(self, client):
        resp = client.get("/datasets/test/opendap.ver")
        assert resp.status_code == 200

    def test_version_content(self, client):
        resp = client.get("/datasets/test/opendap.ver")
        assert "DAP/2.0" in resp.text


class TestHelpRoute:
    def test_help_status_200(self, client):
        resp = client.get("/datasets/test/opendap.help")
        assert resp.status_code == 200
        assert "<html>" in resp.text

    def test_bare_path_returns_help(self, client):
        resp = client.get("/datasets/test/opendap")
        assert resp.status_code == 200
        assert "<html>" in resp.text


class TestErrorHandling:
    def test_bad_constraint_returns_dap_error(self, client):
        resp = client.get("/datasets/test/opendap.dds?temp[abc]")
        assert resp.status_code == 400
        assert "Error {" in resp.text
        assert "code = 1000" in resp.text

    def test_variable_not_found_returns_dap_error(self, client):
        resp = client.get("/datasets/test/opendap.dds?nonexistent")
        assert resp.status_code == 400
        assert "Error {" in resp.text
        assert "code = 1002" in resp.text

    def test_index_out_of_range_returns_dap_error(self, client):
        resp = client.get("/datasets/test/opendap.dds?temp[0:999][0:0][0:0]")
        assert resp.status_code == 400
        assert "Error {" in resp.text
        assert "code = 1003" in resp.text


class TestMemoryThreshold:
    def test_request_too_large(self):
        """Plugin with very low memory limit should reject large requests."""
        ds = xr.Dataset(
            {
                "big": xr.DataArray(
                    np.zeros((100, 100), dtype="float64"),
                    dims=["y", "x"],
                ),
            },
            coords={
                "y": np.arange(100, dtype="float64"),
                "x": np.arange(100, dtype="float64"),
            },
        )
        # Set a very low limit (1 byte)
        plugin = OpenDapPlugin(max_request_memory_bytes=1)
        rest = xpublish.Rest({"test": ds}, plugins={"opendap": plugin})
        client = TestClient(rest.app)

        resp = client.get("/datasets/test/opendap.dods")
        assert resp.status_code == 413
        assert "Error {" in resp.text
        assert "code = 1004" in resp.text


class TestDASWithConstraints:
    def test_das_with_dap2_constraint(self, client):
        resp = client.get("/datasets/test/opendap.das?temp[0:1][0:1][0:1]")
        assert resp.status_code == 200
        assert "Attributes {" in resp.text

    def test_das_single_var_projection(self, client):
        resp = client.get("/datasets/test/opendap.das?temp")
        assert resp.status_code == 200
        text = resp.text
        assert "Attributes {" in text
        # temp attributes should be present
        assert 'String units "kelvin"' in text

    def test_das_constraint_invalid(self, client):
        resp = client.get("/datasets/test/opendap.das?nonexistent[0:0][0:0][0:0]")
        assert resp.status_code == 400
        assert "Error {" in resp.text


class TestMultiVarProjection:
    def test_dds_multi_var(self, client):
        resp = client.get("/datasets/test/opendap.dds?temp,time")
        assert resp.status_code == 200
        text = resp.text
        assert "temp" in text
        assert "time" in text

    def test_dods_multi_var(self, client):
        resp = client.get("/datasets/test/opendap.dods?temp,time")
        assert resp.status_code == 200
        assert b"\nData:\n" in resp.content
        binary = resp.content.split(b"\nData:\n")[1]
        assert len(binary) > 0

    def test_dmr_multi_var(self, client):
        resp = client.get("/datasets/test/opendap.dmr?dap4.ce=/temp;/time")
        assert resp.status_code == 200
        root = ET.fromstring(resp.text)
        all_typed_els = []
        for type_name in ("Float64", "Float32", "Int32", "Int64", "UInt8"):
            all_typed_els.extend(root.findall(f"{{{DAP4_NS}}}{type_name}"))
        var_names = {el.get("name") for el in all_typed_els}
        assert "temp" in var_names
        assert "time" in var_names


class TestStrideAndCoordRoutes:
    def test_dods_with_stride(self, client):
        resp = client.get("/datasets/test/opendap.dods?temp[0:2:2][0:1:3][0:1:4]")
        assert resp.status_code == 200
        assert b"\nData:\n" in resp.content

    def test_dods_coord_only(self, client):
        resp = client.get("/datasets/test/opendap.dods?time")
        assert resp.status_code == 200
        dds_text = resp.content.split(b"\nData:\n")[0].decode("utf-8")
        assert "time" in dds_text

    def test_dap4_with_stride(self, client):
        resp = client.get(
            "/datasets/test/opendap.dap?dap4.ce=/temp[0:2:2][0:1:3][0:1:4]",
        )
        assert resp.status_code == 200


class TestConflictingSlicesRoute:
    def test_dods_conflicting_slices(self, client):
        # Same var projected twice with different slices — should succeed (last-wins)
        resp = client.get(
            "/datasets/test/opendap.dods?temp[0:1][0:1][0:1],temp[0:0][0:0][0:0]",
        )
        assert resp.status_code == 200

    def test_dap4_conflicting_slices(self, client):
        resp = client.get(
            "/datasets/test/opendap.dap?dap4.ce=/temp[0:1][0:1][0:1];/temp[0:0][0:0][0:0]",
        )
        assert resp.status_code == 200


class TestUnexpectedExceptionHandling:
    """Test that unexpected exceptions in handlers return proper error responses."""

    @pytest.fixture
    def err_client(self):
        ds = xr.Dataset(
            {
                "temp": xr.DataArray(
                    np.arange(6, dtype="float64").reshape(2, 3),
                    dims=["y", "x"],
                ),
            },
            coords={
                "y": np.arange(2, dtype="float64"),
                "x": np.arange(3, dtype="float64"),
            },
        )
        rest = xpublish.Rest({"test": ds}, plugins={"opendap": OpenDapPlugin()})
        return TestClient(rest.app)

    def test_dds_unexpected_error(self, err_client):
        with patch(
            "xpublish_opendap.plugin.generate_dds",
            side_effect=RuntimeError("boom"),
        ):
            resp = err_client.get("/datasets/test/opendap.dds")
        assert resp.status_code == 500
        assert "Error {" in resp.text

    def test_das_unexpected_error(self, err_client):
        with patch(
            "xpublish_opendap.plugin.generate_das",
            side_effect=RuntimeError("boom"),
        ):
            resp = err_client.get("/datasets/test/opendap.das")
        assert resp.status_code == 500
        assert "Error {" in resp.text

    def test_dods_unexpected_error(self, err_client):
        with patch(
            "xpublish_opendap.plugin.generate_dods",
            side_effect=RuntimeError("boom"),
        ):
            resp = err_client.get("/datasets/test/opendap.dods")
        assert resp.status_code == 500
        assert "Error {" in resp.text

    def test_dmr_unexpected_error(self, err_client):
        with patch(
            "xpublish_opendap.plugin.generate_dmr",
            side_effect=RuntimeError("boom"),
        ):
            resp = err_client.get("/datasets/test/opendap.dmr")
        assert resp.status_code == 500
        assert "Error" in resp.text

    def test_dap4_data_unexpected_error(self, err_client):
        with patch(
            "xpublish_opendap.plugin.generate_dap4_data",
            side_effect=RuntimeError("boom"),
        ):
            resp = err_client.get("/datasets/test/opendap.dap")
        assert resp.status_code == 500
        assert "Error" in resp.text
