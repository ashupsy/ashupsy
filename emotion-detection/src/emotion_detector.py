"""Core face detection + emotion recognition engine.

This module wraps two pieces:

1. **Face detection** using OpenCV's Haar cascade classifier (fast, no GPU
   needed, ships with OpenCV).
2. **Emotion classification** using a small Keras CNN (a "mini-Xception"
   network trained on the FER-2013 dataset). The model predicts one of
   seven emotions for each detected face.

The design keeps the two stages independent so you can swap either one out
(e.g. use a DNN face detector, or a different emotion model) without
touching the rest of the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# The seven emotion classes the FER-2013 model was trained on, in the
# exact index order the network outputs them.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# A colour (BGR) per emotion, used when drawing labels/boxes on frames.
EMOTION_COLORS = {
    "Angry": (60, 60, 220),
    "Disgust": (60, 160, 60),
    "Fear": (150, 60, 150),
    "Happy": (40, 200, 240),
    "Sad": (200, 120, 40),
    "Surprise": (40, 220, 220),
    "Neutral": (200, 200, 200),
}

# Location of the pretrained emotion model relative to the project root.
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(_MODEL_DIR, "fer2013_mini_XCEPTION.102-0.66.hdf5")


@dataclass
class Detection:
    """A single detected face and its predicted emotion."""

    box: tuple[int, int, int, int]  # (x, y, w, h)
    emotion: str
    confidence: float
    scores: dict[str, float]  # probability for every emotion class


class EmotionDetector:
    """Detects faces in an image and classifies each face's emotion."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        cascade_path: str | None = None,
        min_face_size: tuple[int, int] = (48, 48),
    ) -> None:
        self.min_face_size = min_face_size

        # --- Face detector -------------------------------------------------
        # OpenCV ships the frontal-face Haar cascade with the package, so we
        # can locate it via cv2.data without shipping our own copy.
        if cascade_path is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")

        # --- Emotion model -------------------------------------------------
        # Imported lazily so that face-only use (or environments without
        # TensorFlow) still works, and so the (heavy) import cost is only
        # paid when a model is actually requested.
        self.model = None
        self._model_input_size = (64, 64)
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        from tensorflow.keras.models import load_model  # noqa: PLC0415

        self.model = load_model(model_path, compile=False)
        # Infer the spatial input size the model expects (height, width).
        _, h, w, _ = self.model.input_shape
        self._model_input_size = (int(w), int(h))

    @property
    def has_emotion_model(self) -> bool:
        return self.model is not None

    # ---------------------------------------------------------------------
    # Detection
    # ---------------------------------------------------------------------
    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes (x, y, w, h) for every face in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [tuple(int(v) for v in face) for face in faces]

    def classify_emotion(
        self, gray_face: np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Classify a single, already-cropped grayscale face ROI."""
        if self.model is None:
            raise RuntimeError(
                "No emotion model loaded. Run download_model.py first, or "
                "pass a valid model_path to EmotionDetector."
            )

        # Preprocess exactly the way the FER-2013 model was trained:
        # resize -> scale to [0, 1] -> shape (1, H, W, 1).
        roi = cv2.resize(gray_face, self._model_input_size)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(np.expand_dims(roi, -1), 0)

        preds = self.model.predict(roi, verbose=0)[0]
        scores = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
        top_idx = int(np.argmax(preds))
        return EMOTIONS[top_idx], float(preds[top_idx]), scores

    def analyze(self, frame: np.ndarray) -> list[Detection]:
        """Detect all faces in a BGR frame and classify each one's emotion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections: list[Detection] = []

        for (x, y, w, h) in self.detect_faces(frame):
            if self.model is None:
                detections.append(
                    Detection((x, y, w, h), "Face", 1.0, {})
                )
                continue

            face_roi = gray[y : y + h, x : x + w]
            if face_roi.size == 0:
                continue
            emotion, confidence, scores = self.classify_emotion(face_roi)
            detections.append(Detection((x, y, w, h), emotion, confidence, scores))

        return detections

    # ---------------------------------------------------------------------
    # Drawing helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw boxes and emotion labels onto a copy-free frame in place."""
        for det in detections:
            x, y, w, h = det.box
            color = EMOTION_COLORS.get(det.emotion, (0, 255, 0))
            label = (
                f"{det.emotion} {det.confidence * 100:.0f}%"
                if det.scores
                else det.emotion
            )

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Filled label background for readability.
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, (x, y - th - 10), (x + tw + 6, y), color, cv2.FILLED
            )
            cv2.putText(
                frame,
                label,
                (x + 3, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        return frame
