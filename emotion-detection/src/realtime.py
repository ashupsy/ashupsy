"""Real-time emotion detection from a webcam.

Opens the default camera, runs face detection + emotion classification on
every frame, and shows the annotated video in a window.

Usage:
    python -m src.realtime                # use default webcam (index 0)
    python -m src.realtime --camera 1     # pick a different camera
    python -m src.realtime --no-emotion   # face boxes only (no model needed)

Controls:
    q  or  Esc   quit
    s            save a screenshot of the current frame
"""

from __future__ import annotations

import argparse
import time

import cv2

from src.emotion_detector import DEFAULT_MODEL_PATH, EmotionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time emotion detection")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the emotion model (.hdf5)",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="Detect faces only, skip emotion classification",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally flip the frame (selfie view)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = "" if args.no_emotion else args.model
    detector = EmotionDetector(model_path=model_path)

    if not args.no_emotion and not detector.has_emotion_model:
        print(
            "No emotion model found. Run `python download_model.py` first, "
            "or pass --no-emotion for face detection only."
        )
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    print("Running. Press 'q' or Esc to quit, 's' to save a screenshot.")
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

            detections = detector.analyze(frame)
            detector.draw(frame, detections)

            # Rolling FPS estimate.
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(
                frame,
                f"{fps:.1f} FPS  |  {len(detections)} face(s)",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Emotion Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
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
