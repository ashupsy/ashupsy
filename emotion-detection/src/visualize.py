"""Drawing helpers shared by the real-time and image CLIs.

Renders a :class:`FaceResult` onto a BGR frame: bounding box, top emotion,
valence/arousal meters (when present) and the active FACS Action Units (when
present).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.backends.base import FaceResult

FONT = cv2.FONT_HERSHEY_SIMPLEX

EMOTION_COLORS = {
    "Anger": (60, 60, 220), "anger": (60, 60, 220),
    "Disgust": (60, 160, 60), "disgust": (60, 160, 60),
    "Fear": (150, 60, 150), "fear": (150, 60, 150),
    "Happiness": (40, 200, 240), "happiness": (40, 200, 240), "Happy": (40, 200, 240),
    "Sadness": (200, 120, 40), "sadness": (200, 120, 40), "Sad": (200, 120, 40),
    "Surprise": (40, 220, 220), "surprise": (40, 220, 220),
    "Neutral": (200, 200, 200), "neutral": (200, 200, 200),
    "Contempt": (120, 90, 200), "contempt": (120, 90, 200),
}


def _color(emotion: str) -> tuple[int, int, int]:
    return EMOTION_COLORS.get(emotion, (0, 220, 0))


def _draw_meter(frame, x, y, width, value, label, lo=-1.0, hi=1.0):
    """Draw a horizontal meter for a value in [lo, hi] with a centre mark."""
    height = 8
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 70), cv2.FILLED)
    frac = (max(lo, min(hi, value)) - lo) / (hi - lo)
    fill = int(width * frac)
    col = (60, 200, 60) if value >= 0 else (60, 60, 200)
    cv2.rectangle(frame, (x, y), (x + fill, y + height), col, cv2.FILLED)
    # centre (zero) tick
    cx = x + width // 2
    cv2.line(frame, (cx, y - 1), (cx, y + height + 1), (220, 220, 220), 1)
    cv2.putText(frame, f"{label}:{value:+.2f}", (x + width + 6, y + height),
                FONT, 0.4, (230, 230, 230), 1, cv2.LINE_AA)


def draw_face(frame: np.ndarray, det: FaceResult, show_aus: bool = True) -> None:
    x, y, w, h = det.box
    color = _color(det.emotion)

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Emotion label (with confidence if we have per-class scores)
    conf = det.emotion_scores.get(det.emotion) if det.emotion_scores else None
    label = f"{det.emotion} {conf * 100:.0f}%" if conf is not None else det.emotion
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, cv2.FILLED)
    cv2.putText(frame, label, (x + 3, y - 6), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    # Valence / arousal meters below the box
    ry = y + h + 12
    if det.valence is not None:
        _draw_meter(frame, x, ry, min(120, w), det.valence, "V")
        ry += 16
    if det.arousal is not None:
        _draw_meter(frame, x, ry, min(120, w), det.arousal, "A")
        ry += 16

    # Active Action Units, listed to the right of the box
    if show_aus and det.action_units:
        active = sorted(
            det.active_action_units(threshold=0.5).items(),
            key=lambda kv: -kv[1],
        )
        ty = y
        for au, val in active[:8]:
            cv2.putText(frame, f"{au} {val:.2f}", (x + w + 8, ty + 12),
                        FONT, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
            ty += 15
