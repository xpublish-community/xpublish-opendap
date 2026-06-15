"""xpublish_opendap provides an OpenDAP router for Xpublish."""

from xpublish_opendap.plugin import OpenDapPlugin

__all__ = ["OpenDapPlugin", "__version__"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
