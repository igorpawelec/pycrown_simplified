"""
PyCrown Simplified — Individual tree crown segmentation from CHM.

Copyright (C) 2025 Igor Pawelec
Licence: GPLv3
"""

try:
    from importlib.metadata import version
    __version__ = version("pycrown-simplified")
except Exception:
    __version__ = "0.1.0"

# ── Import strategy ──────────────────────────────────────────────
# Try eager import first. Fall back to lazy if deps are broken.

try:
    from .pycrown import PyCrown
    _LAZY_MODE = False
except (ImportError, OSError) as _init_err:
    _LAZY_MODE = True

    def __getattr__(name):
        if name == "PyCrown":
            try:
                from .pycrown import PyCrown
            except (ImportError, OSError) as e:
                raise ImportError(
                    f"Cannot load PyCrown: {e}\n"
                    f"Install deps: conda install -c conda-forge "
                    f"numpy numba rasterio scipy scikit-image fiona"
                ) from e
            globals()["PyCrown"] = PyCrown
            return PyCrown
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["PyCrown"]
