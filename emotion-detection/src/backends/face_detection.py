"""Shared OpenCV face detector.

The HSEmotion / FER backends only classify emotion on an already-cropped
face, so they need a face detector in front of them. We use OpenCV's Haar
cascade (CPU-only, ships with OpenCV) here so all backends share one, cheap,
dependency-free detector. Py-Feat brings its own (RetinaFace) detector and
does not use this.
"""

from __future__ import annotations

import cv2
import numpy as np


class HaarFaceDetector:
    def __init__(self, min_face_size: tuple[int, int] = (48, 48)) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
        self.min_face_size = min_face_size

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [tuple(int(v) for v in f) for f in faces]
