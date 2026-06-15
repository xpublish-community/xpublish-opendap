"""DAP4 HTTP header constants and content types."""

from __future__ import annotations

DAP4_HEADERS: dict[str, str] = {
    "XDAP": "4.0",
    "XOPeNDAP-Server": "xpublish-opendap/2.0",
}

CONTENT_TYPES: dict[str, str] = {
    "dmr": "application/vnd.opendap.dap4.dataset-metadata+xml",
    "dap": "application/vnd.opendap.dap4.data",
    "dsr": "application/vnd.opendap.dap4.dataset-services+xml",
    "error": "application/vnd.opendap.dap4.error+xml",
}
