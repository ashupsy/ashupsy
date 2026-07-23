"""Legacy FER-2013 backend (OpenCV Haar + mini-Xception).

Wraps the original :class:`EmotionDetector` in the shared backend interface so
it can be selected alongside the newer, more granular backends. Reports 7
categorical emotions only — no valence/arousal, no action units.
"""

from __future__ import annotations

import numpy as np

from src.backends.base import EmotionBackend, FaceResult
from src.emotion_detector import DEFAULT_MODEL_PATH, EmotionDetector


class FERBackend(EmotionBackend):
    name = "fer"
    provides_valence_arousal = False
    provides_action_units = False

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.detector = EmotionDetector(model_path=model_path)

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        results: list[FaceResult] = []
        for det in self.detector.analyze(frame_bgr):
            results.append(
                FaceResult(
                    box=det.box,
                    emotion=det.emotion,
                    emotion_scores=det.scores,
                    backend=self.name,
                )
            )
        return results
