"""(Optional) Train the emotion-classification model on FER-2013.

Most users should just run `python download_model.py` to grab a pretrained
model. This script is here for completeness / reproducibility and mirrors the
"mini-Xception" architecture commonly used on the FER-2013 dataset.

Dataset: download `fer2013.csv` from the FER-2013 Kaggle challenge and pass
its path with --data.

Usage:
    python -m src.train --data path/to/fer2013.csv --epochs 100
"""

from __future__ import annotations

import argparse
import os

import numpy as np


IMG_SIZE = 64
NUM_CLASSES = 7


def load_fer2013(csv_path: str):
    """Load the FER-2013 CSV into (X, y) arrays scaled to [0, 1]."""
    import pandas as pd  # noqa: PLC0415

    data = pd.read_csv(csv_path)
    pixels = data["pixels"].tolist()
    faces = []
    for row in pixels:
        face = np.asarray(row.split(" "), dtype="float32").reshape(48, 48)
        import cv2  # noqa: PLC0415

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        faces.append(face)
    x = np.expand_dims(np.asarray(faces), -1) / 255.0
    y = np.eye(NUM_CLASSES)[data["emotion"].to_numpy()]
    return x, y


def build_mini_xception():
    """Build the mini-Xception CNN used for FER-2013 emotion recognition."""
    from tensorflow.keras import layers, models, regularizers  # noqa: PLC0415

    reg = regularizers.l2(0.01)
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(8, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(8, 3, padding="same", use_bias=False,
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for filters in (16, 32, 64, 128):
        residual = layers.Conv2D(filters, 1, strides=2, padding="same",
                                 use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(filters, 3, padding="same",
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same",
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.add([x, residual])

    x = layers.Conv2D(NUM_CLASSES, 3, padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax")(x)

    return models.Model(inputs, outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train FER-2013 emotion model")
    parser.add_argument("--data", required=True, help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "models", "emotion_model.hdf5"),
    )
    args = parser.parse_args()

    print("Loading dataset...")
    x, y = load_fer2013(args.data)
    split = int(len(x) * 0.9)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_mini_xception()
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    from tensorflow.keras.callbacks import (  # noqa: PLC0415
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    callbacks = [
        ModelCheckpoint(args.output, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          verbose=1),
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )
    print(f"Best model saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
