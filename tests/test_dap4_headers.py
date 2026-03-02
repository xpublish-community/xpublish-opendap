# ruff: noqa: D100,D101,D102,D103
"""Tests for dap/dap4/headers.py — DAP4 header constants."""

from xpublish_opendap.dap.dap4.headers import CONTENT_TYPES, DAP4_HEADERS


class TestDAP4HeaderConstants:
    def test_xdap_version(self):
        assert DAP4_HEADERS['XDAP'] == '4.0'

    def test_server_header_contains_xpublish(self):
        assert 'xpublish-opendap' in DAP4_HEADERS['XOPeNDAP-Server']

    def test_dmr_content_type(self):
        assert CONTENT_TYPES['dmr'] == 'application/vnd.opendap.dap4.dataset-metadata+xml'

    def test_dap_content_type(self):
        assert CONTENT_TYPES['dap'] == 'application/vnd.opendap.dap4.data'

    def test_dsr_content_type(self):
        assert CONTENT_TYPES['dsr'] == 'application/vnd.opendap.dap4.dataset-services+xml'

    def test_error_content_type(self):
        assert CONTENT_TYPES['error'] == 'application/vnd.opendap.dap4.error+xml'

    def test_content_type_keys(self):
        assert set(CONTENT_TYPES.keys()) == {'dmr', 'dap', 'dsr', 'error'}
