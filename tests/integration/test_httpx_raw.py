"""Raw HTTP tests using httpx to validate DAP2 and DAP4 protocol-level correctness."""

import xml.etree.ElementTree as ET

import httpx
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# DAP2 endpoints
# ---------------------------------------------------------------------------


class TestDAP2Raw:
    """DAP2 protocol-level HTTP tests."""

    def test_dds_status_and_content_type(self, opendap_base):
        """GET .dds returns 200 with text/plain content type."""
        r = httpx.get(f"{opendap_base}.dds")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_das_status_and_content_type(self, opendap_base):
        """GET .das returns 200 with text/plain content type."""
        r = httpx.get(f"{opendap_base}.das")
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]

    def test_dods_status_and_content_type(self, opendap_base):
        """GET .dods returns 200 with application/octet-stream content type."""
        r = httpx.get(f"{opendap_base}.dods")
        assert r.status_code == 200
        assert "application/octet-stream" in r.headers["content-type"]

    def test_dds_with_constraint(self, opendap_base):
        """GET .dds with constraint returns subset of structure."""
        r = httpx.get(f"{opendap_base}.dds?air[0:0][0:0][0:0]")
        assert r.status_code == 200
        body = r.text
        assert "air" in body

    def test_dap2_headers(self, opendap_base):
        """DAP2 responses include XDODS-Server header."""
        r = httpx.get(f"{opendap_base}.dds")
        assert "XDODS-Server" in r.headers or "xdods-server" in r.headers

    def test_version_endpoint(self, opendap_base):
        """GET .ver returns 200."""
        r = httpx.get(f"{opendap_base}.ver")
        assert r.status_code == 200

    def test_help_endpoint(self, opendap_base):
        """GET .help returns 200 with HTML content."""
        r = httpx.get(f"{opendap_base}.help")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ---------------------------------------------------------------------------
# DAP4 endpoints
# ---------------------------------------------------------------------------


class TestDAP4Raw:
    """DAP4 protocol-level HTTP tests."""

    def test_dmr_status_and_content_type(self, opendap_base):
        """GET .dmr returns 200 with dataset-metadata+xml content type."""
        r = httpx.get(f"{opendap_base}.dmr")
        assert r.status_code == 200
        assert "dataset-metadata+xml" in r.headers["content-type"]

    def test_dap4_data_status_and_content_type(self, opendap_base):
        """GET .dap returns 200 with dap4.data content type."""
        r = httpx.get(f"{opendap_base}.dap")
        assert r.status_code == 200
        assert "dap4.data" in r.headers["content-type"]

    def test_dsr_status_and_content_type(self, opendap_base):
        """GET .dsr returns 200 with dataset-services+xml content type."""
        r = httpx.get(f"{opendap_base}.dsr")
        assert r.status_code == 200
        assert "dataset-services+xml" in r.headers["content-type"]

    def test_dmr_valid_xml(self, opendap_base):
        """DMR response is valid XML with expected root element."""
        r = httpx.get(f"{opendap_base}.dmr")
        root = ET.fromstring(r.text)
        # Root tag should be Dataset (possibly namespaced)
        local_name = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        assert local_name == "Dataset"

    def test_dmr_with_constraint(self, opendap_base):
        """GET .dmr with dap4.ce constraint returns filtered DMR."""
        r = httpx.get(f"{opendap_base}.dmr", params={"dap4.ce": "/air"})
        assert r.status_code == 200
        body = r.text
        assert "air" in body

    def test_dap4_headers(self, opendap_base):
        """DAP4 responses include XDAP: 4.0 header."""
        r = httpx.get(f"{opendap_base}.dmr")
        xdap = r.headers.get("XDAP") or r.headers.get("xdap")
        assert xdap == "4.0"

    def test_dap4_data_starts_with_dmr(self, opendap_base):
        """DAP4 .dap response body starts with XML declaration (DMR preamble)."""
        r = httpx.get(f"{opendap_base}.dap")
        # The binary response begins with a DMR XML section
        assert r.content[:5] == b"<?xml"
