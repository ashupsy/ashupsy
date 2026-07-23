"""HSEmotion / EmotiEffLib backend — fast dimensional emotion recognition.

Uses EmotiEffLib (ex-HSEmotion), an EfficientNet trained on AffectNet. With a
``*_va_mtl`` model it returns, per face:

- 8 emotion categories (Anger, Contempt, Disgust, Fear, Happiness, Neutral,
  Sadness, Surprise), and
- continuous **valence** (negative ↔ positive) and **arousal**
  (calm ↔ excited), each roughly in [-1, 1].

EmotiEffLib does not detect faces, so we run OpenCV's Haar detector first and
feed it the cropped faces.

Install:  pip install emotiefflib onnxruntime      # or: emotiefflib[torch]
"""

from __future__ import annotations

import cv2
import numpy as np

from src.backends.base import EmotionBackend, FaceResult
from src.backends.face_detection import HaarFaceDetector

# The 8-class label order EmotiEffLib uses for non-"_7" models.
_EMOTIONS_8 = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]


class HSEmotionBackend(EmotionBackend):
    name = "hsemotion"
    provides_valence_arousal = True
    provides_action_units = False

    def __init__(
        self,
        model_name: str = "enet_b0_8_va_mtl",
        engine: str = "onnx",
        device: str = "cpu",
    ) -> None:
        # Imported lazily so the package is only required when this backend
        # is actually selected.
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # noqa: PLC0415

        self.model_name = model_name
        self._is_mtl = "_mtl" in model_name  # va_mtl models also output valence/arousal
        self.recognizer = EmotiEffLibRecognizer(
            engine=engine, model_name=model_name, device=device
        )
        self.face_detector = HaarFaceDetector()

    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        results: list[FaceResult] = []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in self.face_detector.detect(frame_bgr):
            crop = rgb[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            # logits=False -> the 8 emotion columns come back as probabilities.
            labels, scores = self.recognizer.predict_emotions(crop, logits=False)
            row = np.asarray(scores)[0] if np.ndim(scores) == 2 else np.asarray(scores)

            emotion_scores = {
                _EMOTIONS_8[i]: float(row[i]) for i in range(len(_EMOTIONS_8))
            }
            valence = arousal = None
            if self._is_mtl and len(row) >= len(_EMOTIONS_8) + 2:
                # For MTL models the final two columns are valence, arousal.
                valence = float(row[-2])
                arousal = float(row[-1])

            results.append(
                FaceResult(
                    box=(x, y, w, h),
                    emotion=labels[0],
                    emotion_scores=emotion_scores,
                    valence=valence,
                    arousal=arousal,
                    backend=self.name,
                )
            )
        return results
