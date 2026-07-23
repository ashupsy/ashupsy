"""Py-Feat backend — research-grade, muscle-level (FACS) analysis.

Py-Feat detects faces (RetinaFace) and, per face, reports:

- **Action Units** — ~20 FACS facial-muscle activations (e.g. AU06 cheek
  raiser, AU12 lip-corner puller). This is the most granular signal here and
  the standard used in affective-science research.
- **Categorical emotions** — happiness, sadness, anger, fear, surprise,
  disgust, neutral.
- **Valence & arousal** — continuous, when the loaded model provides them
  (Py-Feat's v2 multitask model does; the classic pipeline does not).

Py-Feat is heavier and slower than HSEmotion (it runs several models per
face), so it targets images / recorded video rather than high-FPS webcam.

Install:  pip install py-feat
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from src.backends.base import EmotionBackend, FaceResult


class PyFeatBackend(EmotionBackend):
    name = "pyfeat"
    provides_valence_arousal = True  # depends on the loaded model; detected at runtime
    provides_action_units = True

    def __init__(self, device: str = "cpu", detector=None) -> None:
        # Imported lazily: Py-Feat (and torch) are only needed when this
        # backend is selected.
        if detector is None:
            from feat import Detector  # noqa: PLC0415

            detector = Detector(device=device)
        self.detector = detector

    # ---------------------------------------------------------------------
    # Primary entry point for files (most efficient — no re-encoding).
    # ---------------------------------------------------------------------
    def analyze_file(self, image_path: str) -> list[FaceResult]:
        fex = self.detector.detect([image_path], data_type="image")
        return self._fex_to_results(fex)

    def analyze_video(self, video_path: str, skip_frames: int | None = None):
        """Return the raw Py-Feat Fex DataFrame for a video (per-frame rows).

        Kept as the rich Fex object so callers can export CSV, plot AU
        timelines, aggregate over frames, etc.
        """
        return self.detector.detect(
            [video_path], data_type="video", skip_frames=skip_frames
        )

    # ---------------------------------------------------------------------
    # Uniform array interface (writes a temp file so Py-Feat can ingest it).
    # ---------------------------------------------------------------------
    def analyze(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, frame_bgr)
            return self.analyze_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------------------------------------------------------------------
    # Convert a Py-Feat Fex DataFrame into our uniform FaceResult list.
    # ---------------------------------------------------------------------
    def _fex_to_results(self, fex) -> list[FaceResult]:
        results: list[FaceResult] = []
        if fex is None or len(fex) == 0:
            return results

        emotions = fex.emotions
        aus = fex.aus
        boxes = fex.faceboxes
        has_va = "valence" in fex.columns and "arousal" in fex.columns

        for i in range(len(fex)):
            emo_row = emotions.iloc[i]
            top_emotion = str(emo_row.idxmax())
            emotion_scores = {str(k): float(v) for k, v in emo_row.items()}

            au_row = aus.iloc[i]
            action_units = {str(k): float(v) for k, v in au_row.items()}

            box_row = boxes.iloc[i]
            x = int(round(float(box_row["FaceRectX"])))
            y = int(round(float(box_row["FaceRectY"])))
            w = int(round(float(box_row["FaceRectWidth"])))
            h = int(round(float(box_row["FaceRectHeight"])))

            valence = float(fex.iloc[i]["valence"]) if has_va else None
            arousal = float(fex.iloc[i]["arousal"]) if has_va else None

            results.append(
                FaceResult(
                    box=(x, y, w, h),
                    emotion=top_emotion,
                    emotion_scores=emotion_scores,
                    valence=valence,
                    arousal=arousal,
                    action_units=action_units,
                    backend=self.name,
                )
            )
        return results
