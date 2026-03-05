"""
PyCrown Simplified — Individual tree crown segmentation from CHM.

Copyright (C) 2025 Igor Pawelec
Licence: GPLv3
"""


def __getattr__(name):
    """Lazy import: PyCrown class is loaded only when accessed."""
    if name == "PyCrown":
        from .pycrown import PyCrown
        return PyCrown
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from importlib.metadata import version
    __version__ = version("pycrown-simplified")
except Exception:
    __version__ = "0.1.0"
