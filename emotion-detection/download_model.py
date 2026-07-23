"""Download the pretrained FER-2013 emotion model.

The real-time detector needs a trained emotion-classification model. Rather
than ship a large binary in git, this script fetches the well-known
"mini-Xception" model (trained on FER-2013, ~0.66 validation accuracy) from
the public `oarriaga/face_classification` repository.

Usage:
    python download_model.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

MODEL_URL = (
    "https://github.com/oarriaga/face_classification/raw/master/"
    "trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fer2013_mini_XCEPTION.102-0.66.hdf5")


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb = downloaded / (1024 * 1024)
    sys.stdout.write(f"\r  {pct:3d}%  ({mb:.1f} MB)")
    sys.stdout.flush()


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"Model already present: {MODEL_PATH}")
        return 0

    print(f"Downloading emotion model to {MODEL_PATH}")
    print(f"  source: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _progress)
        sys.stdout.write("\n")
    except Exception as exc:  # noqa: BLE001
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "You can download the file manually from the URL above and place "
            f"it at {MODEL_PATH}",
            file=sys.stderr,
        )
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
