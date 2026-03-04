from .pycrown import PyCrown

try:
    from importlib.metadata import version
    __version__ = version("pycrown-simplified")
except Exception:
    __version__ = "0.1.0"
