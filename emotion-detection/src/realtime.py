"""Real-time emotion detection from a webcam.

Opens the default camera and runs a chosen backend on every frame, drawing
the emotion, valence/arousal, and (if the backend provides them) Action Units.

Usage:
    python -m src.realtime                       # HSEmotion (default, fast)
    python -m src.realtime --backend fer         # legacy FER-2013 model
    python -m src.realtime --backend pyfeat      # Py-Feat (slow; not ideal live)
    python -m src.realtime --camera 1 --mirror

Controls:
    q / Esc   quit
    s         save a screenshot

Note: HSEmotion is the recommended real-time backend. Py-Feat runs several
models per face and will be only a few FPS on CPU.
"""

from __future__ import annotations

import argparse
import time

import cv2

from src.backends import BACKENDS, get_backend
from src.visualize import draw_face


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time emotion detection")
    p.add_argument("--backend", choices=BACKENDS, default="hsemotion",
                   help="Which recognition backend to use (default: hsemotion)")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--mirror", action="store_true", help="Selfie / mirrored view")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        backend = get_backend(args.backend)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not initialise backend '{args.backend}': {exc}")
        print("Check the install notes in the README for this backend.")
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    print(f"Backend: {args.backend}. Press 'q'/Esc to quit, 's' to screenshot.")
    prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            detections = backend.analyze(frame)
            for det in detections:
                draw_face(frame, det)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(
                frame,
                f"{args.backend} | {fps:.1f} FPS | {len(detections)} face(s)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow("Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                fname = f"screenshot_{int(now)}.png"
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
