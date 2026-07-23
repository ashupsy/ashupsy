"""Deep emotion analysis of an image or video, with CSV export.

Runs a chosen backend over a photo or a recorded video and reports, per face,
the categorical emotion, valence/arousal, and (with Py-Feat) FACS Action
Units. Writes an annotated image and/or a CSV of the results.

Usage:
    # Rich single-image analysis (Action Units + emotions + valence/arousal)
    python -m src.analyze photo.jpg
    python -m src.analyze photo.jpg --backend hsemotion
    python -m src.analyze photo.jpg --csv results.csv --output annotated.jpg

    # Video (Py-Feat writes a per-frame CSV directly)
    python -m src.analyze clip.mp4 --csv clip_fex.csv --skip-frames 5

Backends: pyfeat (default; most granular), hsemotion, fer.
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2

from src.backends import BACKENDS, FaceResult, get_backend
from src.visualize import draw_face

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep emotion analysis of image/video")
    p.add_argument("input", help="Path to an image or video file")
    p.add_argument("--backend", choices=BACKENDS, default="pyfeat",
                   help="Recognition backend (default: pyfeat)")
    p.add_argument("--csv", default=None, help="Write results to this CSV file")
    p.add_argument("--output", default=None,
                   help="Annotated image output path (images only)")
    p.add_argument("--no-show", action="store_true", help="Do not open a window")
    p.add_argument("--skip-frames", type=int, default=None,
                   help="Process every Nth frame (video only, Py-Feat)")
    return p.parse_args()


def _print_face(i: int, det: FaceResult) -> None:
    print(f"\nFace {i}  box={det.box}  ->  {det.summary()}")
    if det.emotion_scores:
        for emo, score in sorted(det.emotion_scores.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(score * 30)
            print(f"    {emo:<10} {score * 100:5.1f}%  {bar}")
    active = det.active_action_units()
    if active:
        print("    Active Action Units:")
        for au, val in sorted(active.items(), key=lambda kv: -kv[1]):
            print(f"      {au:<6} {val:.2f}")


def _result_row(det: FaceResult, frame: int | None = None) -> dict:
    x, y, w, h = det.box
    row = {
        "frame": frame if frame is not None else 0,
        "backend": det.backend,
        "x": x, "y": y, "w": w, "h": h,
        "emotion": det.emotion,
        "valence": det.valence,
        "arousal": det.arousal,
    }
    for emo, score in det.emotion_scores.items():
        row[f"emo_{emo}"] = score
    for au, val in det.action_units.items():
        row[au] = val
    return row


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        print("No rows to write to CSV.")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {path}")


def analyze_image(args, backend) -> int:
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Could not read image: {args.input}")
        return 1

    # Py-Feat can ingest the file directly (avoids a re-encode).
    if hasattr(backend, "analyze_file"):
        detections = backend.analyze_file(args.input)
    else:
        detections = backend.analyze(frame)

    if not detections:
        print("No faces detected.")
    for i, det in enumerate(detections, 1):
        _print_face(i, det)
        draw_face(frame, det)

    output = args.output or (
        os.path.splitext(args.input)[0] + "_annotated" +
        (os.path.splitext(args.input)[1] or ".png")
    )
    cv2.imwrite(output, frame)
    print(f"\nAnnotated image saved to {output}")

    if args.csv:
        _write_csv(args.csv, [_result_row(d) for d in detections])

    if not args.no_show:
        cv2.imshow("Emotion Analysis", frame)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def analyze_video(args, backend) -> int:
    # Py-Feat has native, efficient video handling and rich CSV output.
    if args.backend == "pyfeat":
        print("Running Py-Feat on video (this can take a while)...")
        fex = backend.analyze_video(args.input, skip_frames=args.skip_frames)
        csv_path = args.csv or (os.path.splitext(args.input)[0] + "_fex.csv")
        fex.to_csv(csv_path, index=False)
        print(f"Per-frame results written to {csv_path} ({len(fex)} rows)")
        return 0

    # Other backends: iterate frames ourselves.
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Could not open video: {args.input}")
        return 1
    rows: list[dict] = []
    frame_idx = 0
    step = args.skip_frames or 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                for det in backend.analyze(frame):
                    rows.append(_result_row(det, frame=frame_idx))
            frame_idx += 1
    finally:
        cap.release()
    print(f"Processed {frame_idx} frame(s), {len(rows)} face detection(s).")
    if args.csv:
        _write_csv(args.csv, rows)
    return 0


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        return 1

    try:
        backend = get_backend(args.backend)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not initialise backend '{args.backend}': {exc}")
        print("See the README install notes for this backend.")
        return 1

    ext = os.path.splitext(args.input)[1].lower()
    if ext in _VIDEO_EXTS:
        return analyze_video(args, backend)
    return analyze_image(args, backend)


if __name__ == "__main__":
    raise SystemExit(main())
