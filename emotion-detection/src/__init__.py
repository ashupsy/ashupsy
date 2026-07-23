"""Face detection + emotion recognition package.

Backends live in :mod:`src.backends`; construct one with
``src.backends.get_backend("hsemotion" | "pyfeat" | "fer")``.
"""

from src.backends import BACKENDS, FaceResult, get_backend

__all__ = ["get_backend", "BACKENDS", "FaceResult"]
