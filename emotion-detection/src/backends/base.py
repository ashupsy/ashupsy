"""Shared types and interface for emotion-recognition backends.

Every backend takes a BGR image frame and returns a list of `FaceResult`,
one per detected face. Backends differ in how much they report:

- **categorical** — a top emotion + per-class scores (all backends)
- **dimensional** — continuous valence & arousal (HSEmotion, Py-Feat)
- **action units** — FACS facial-muscle activations (Py-Feat only)

Fields a backend can't produce are left as ``None`` / empty, so downstream
code (drawing, CSV export) can treat every backend uniformly and simply
skip what isn't there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceResult:
    """Everything a backend can say about one detected face."""

    box: tuple[int, int, int, int]  # (x, y, w, h) in pixels

    # Categorical emotion
    emotion: str = "Unknown"
    emotion_scores: dict[str, float] = field(default_factory=dict)

    # Dimensional affect (continuous). Ranges follow each backend's
    # convention; both HSEmotion and Py-Feat use roughly [-1, 1] for
    # valence and arousal.
    valence: float | None = None
    arousal: float | None = None

    # FACS Action Units: {"AU06": 0.83, "AU12": 0.91, ...}
    action_units: dict[str, float] = field(default_factory=dict)

    backend: str = ""

    def active_action_units(self, threshold: float = 0.5) -> dict[str, float]:
        """Action units whose intensity exceeds ``threshold``."""
        return {
            au: v for au, v in self.action_units.items() if v is not None and v >= threshold
        }

    def summary(self) -> str:
        """One-line human-readable summary of this face."""
        parts = [self.emotion]
        if self.valence is not None:
            parts.append(f"val={self.valence:+.2f}")
        if self.arousal is not None:
            parts.append(f"aro={self.arousal:+.2f}")
        active = self.active_action_units()
        if active:
            aus = ",".join(sorted(active))
            parts.append(f"AUs[{aus}]")
        return "  ".join(parts)


class EmotionBackend(ABC):
    """Interface every emotion-recognition backend implements."""

    name: str = "base"

    #: What this backend can report — used by the UI/exporters.
    provides_valence_arousal: bool = False
    provides_action_units: bool = False

    @abstractmethod
    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        """Detect faces in a BGR frame and return per-face results."""
        raise NotImplementedError
