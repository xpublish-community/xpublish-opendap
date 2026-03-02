# ruff: noqa: D100,D101,D102,D103
"""Tests for xpublish_opendap/__init__.py."""


class TestVersionImportFallback:
    def test_version_import_fallback(self):
        import importlib
        import sys
        from unittest.mock import patch

        # Force the ImportError path in __init__.py
        saved_version = sys.modules.get('xpublish_opendap._version')
        with patch.dict(sys.modules, {'xpublish_opendap._version': None}):
            # Remove cached module to force re-import
            if 'xpublish_opendap' in sys.modules:
                del sys.modules['xpublish_opendap']
            import xpublish_opendap

            importlib.reload(xpublish_opendap)
            assert xpublish_opendap.__version__ == 'unknown'

        # Restore the module
        if saved_version is not None:
            sys.modules['xpublish_opendap._version'] = saved_version
        if 'xpublish_opendap' in sys.modules:
            del sys.modules['xpublish_opendap']
        import xpublish_opendap  # noqa: F811

        importlib.reload(xpublish_opendap)
