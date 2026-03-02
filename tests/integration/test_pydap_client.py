"""Integration tests using pydap's DAP2 and DAP4 clients."""

import numpy as np
import pytest

try:
    import pydap  # noqa: F401

    HAS_PYDAP = True
except ImportError:
    HAS_PYDAP = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_PYDAP, reason="pydap not installed"),
]


def _open_dap2(url):
    from pydap.client import open_url

    return open_url(url)


class TestPydapDAP2:
    """Tests using pydap.client.open_url (DAP2)."""

    def test_open_dap2(self, opendap_base):
        """Can open the remote dataset via DAP2."""
        ds = _open_dap2(opendap_base)
        assert ds is not None

    def test_dap2_variable_names(self, opendap_base):
        """Expected variable keys are present."""
        ds = _open_dap2(opendap_base)
        assert "air" in ds.keys()
        assert "lat" in ds.keys()
        assert "lon" in ds.keys()
        assert "time" in ds.keys()

    def test_dap2_variable_shape(self, opendap_base):
        """Air variable has the expected shape."""
        ds = _open_dap2(opendap_base)
        assert ds["air"].shape == (2920, 25, 53)

    def test_dap2_read_data(self, opendap_base, reference_ds):
        """A small data slice matches the reference."""
        ds = _open_dap2(opendap_base)
        actual = np.asarray(ds["air"][0:1, 0:1, 0:1].data).squeeze()
        expected = reference_ds["air"].values[0, 0, 0]
        np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)

    def test_dap2_read_coordinate(self, opendap_base, reference_ds):
        """Latitude coordinate values match the reference."""
        ds = _open_dap2(opendap_base)
        actual = np.asarray(ds["lat"][:].data)
        expected = reference_ds["lat"].values
        np.testing.assert_allclose(actual, expected, rtol=1e-5)


class TestPydapDAP4:
    """Tests using pydap.client.open_url with DAP4 protocol.

    These are xfail(strict=False) because pydap's DAP4 client support
    may not be fully compatible with our server's DAP4 encoding.
    """

    @pytest.mark.xfail(strict=False, reason="DAP4 client support may not be compatible")
    def test_open_dap4(self, opendap_base):
        """Can open the remote dataset via DAP4."""
        from pydap.client import open_url

        ds = open_url(opendap_base, protocol="dap4")
        assert ds is not None

    @pytest.mark.xfail(strict=False, reason="DAP4 client support may not be compatible")
    def test_dap4_variable_names(self, opendap_base):
        """Expected variable keys are present in DAP4 dataset."""
        from pydap.client import open_url

        ds = open_url(opendap_base, protocol="dap4")
        assert "air" in ds.keys()

    @pytest.mark.xfail(strict=False, reason="DAP4 client support may not be compatible")
    def test_dap4_read_data(self, opendap_base, reference_ds):
        """A small data slice matches the reference via DAP4."""
        from pydap.client import open_url

        ds = open_url(opendap_base, protocol="dap4")
        actual = np.asarray(ds["air"][0:1, 0:1, 0:1].data).squeeze()
        expected = reference_ds["air"].values[0, 0, 0]
        np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)
