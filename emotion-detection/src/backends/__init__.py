"""Emotion-recognition backends.

Three interchangeable backends behind one interface (:class:`EmotionBackend`,
returning :class:`FaceResult`):

- ``hsemotion`` — fast, real-time. 8 emotions + continuous valence/arousal.
- ``pyfeat``    — research-grade. 20 FACS Action Units + emotions (+ valence/
                  arousal with the v2 model). Heavier; best for images/video.
- ``fer``       — the original OpenCV + FER-2013 model. 7 emotions only.

Use :func:`get_backend` to construct one by name; imports are deferred so you
only need the dependencies of the backend you actually use.
"""

from __future__ import annotations

from src.backends.base import EmotionBackend, FaceResult

BACKENDS = ("hsemotion", "pyfeat", "fer")


def get_backend(name: str, **kwargs) -> EmotionBackend:
    name = name.lower()
    if name == "hsemotion":
        from src.backends.hsemotion_backend import HSEmotionBackend

        return HSEmotionBackend(**kwargs)
    if name == "pyfeat":
        from src.backends.pyfeat_backend import PyFeatBackend

        return PyFeatBackend(**kwargs)
    if name == "fer":
        from src.backends.fer_backend import FERBackend

        return FERBackend(**kwargs)
    raise ValueError(
        f"Unknown backend '{name}'. Choose from: {', '.join(BACKENDS)}"
    )


__all__ = ["EmotionBackend", "FaceResult", "get_backend", "BACKENDS"]
