"""Run emotion detection on a single image file.

Usage:
    python -m src.detect_image path/to/photo.jpg
    python -m src.detect_image photo.jpg --output annotated.jpg
    python -m src.detect_image photo.jpg --no-show     # don't open a window

Prints each detected face's emotion and per-class scores, and writes an
annotated copy of the image.
"""

from __future__ import annotations

import argparse
import os

import cv2

from src.emotion_detector import DEFAULT_MODEL_PATH, EmotionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emotion detection on an image")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH, help="Path to the emotion model"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the annotated image (default: <name>_annotated.<ext>)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open a display window"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return 1

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Could not read image: {args.image}")
        return 1

    detector = EmotionDetector(model_path=args.model)
    detections = detector.analyze(frame)

    if not detections:
        print("No faces detected.")
    for i, det in enumerate(detections, 1):
        print(f"\nFace {i}  box={det.box}  ->  {det.emotion} "
              f"({det.confidence * 100:.1f}%)")
        for emo, score in sorted(det.scores.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(score * 30)
            print(f"    {emo:<9} {score * 100:5.1f}%  {bar}")

    detector.draw(frame, detections)

    output = args.output
    if output is None:
        base, ext = os.path.splitext(args.image)
        output = f"{base}_annotated{ext or '.png'}"
    cv2.imwrite(output, frame)
    print(f"\nAnnotated image saved to {output}")

    if not args.no_show:
        cv2.imshow("Emotion Detection", frame)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
