"""HTTP header constants for DAP2 responses."""

DAP2_HEADERS: dict[str, str] = {
    'XDODS-Server': 'xpublish-opendap/2.0',
    'XOPeNDAP-Server': 'xpublish-opendap/2.0',
}

CONTENT_DESCRIPTIONS: dict[str, str] = {
    'dds': 'dods-dds',
    'das': 'dods-das',
    'dods': 'dods-data',
    'error': 'dods-error',
}

CONTENT_TYPES: dict[str, str] = {
    'dds': 'text/plain',
    'das': 'text/plain',
    'dods': 'application/octet-stream',
    'error': 'text/plain',
    'version': 'text/plain',
    'help': 'text/html',
}
