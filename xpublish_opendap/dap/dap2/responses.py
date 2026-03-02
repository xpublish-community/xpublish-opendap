"""DAP2 Error, Version, and Help response generators."""

from __future__ import annotations

from collections.abc import Iterator


def generate_error(code: int, message: str) -> Iterator[str]:
    """Yield DAP2 Error response text.

    Args:
        code: The DAP error code.
        message: The error message.

    Yields:
        Lines of the error response.
    """
    escaped = message.replace('"', '\\"')
    yield 'Error {\n'
    yield f'    code = {code};\n'
    yield f'    message = "{escaped}";\n'
    yield '};\n'


def generate_version(
    dap_version: str = '2.0',
    server_version: str = 'xpublish-opendap/2.0',
) -> Iterator[str]:
    """Yield DAP2 Version response text.

    Args:
        dap_version: The DAP protocol version.
        server_version: The server version string.

    Yields:
        Lines of the version response.
    """
    yield f'DAP/{dap_version}\n'
    yield f'Server: {server_version}\n'


def generate_help() -> str:
    """Return HTML Help response listing recognized extensions.

    Returns:
        HTML string describing available endpoints.
    """
    return (
        '<html><head><title>OPeNDAP Help</title></head><body>\n'
        '<h1>OPeNDAP Server Help</h1>\n'
        '<p>Supported DAP2 extensions:</p>\n'
        '<ul>\n'
        '<li><code>.dds</code> - Dataset Descriptor Structure</li>\n'
        '<li><code>.das</code> - Dataset Attribute Structure</li>\n'
        '<li><code>.dods</code> - Data (DataDDS)</li>\n'
        '<li><code>.ver</code> - Server version</li>\n'
        '<li><code>.help</code> - This help page</li>\n'
        '</ul>\n'
        '<p>Constraint expressions can be appended as query strings.</p>\n'
        '</body></html>\n'
    )
