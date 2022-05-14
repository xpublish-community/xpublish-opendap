"""
xpublish_opendap provides an OpenDAP router for Xpublish
"""

from xpublish_opendap.router import dap_router

__all__ = ["dap_router"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
