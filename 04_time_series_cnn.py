"""Classify the raw 128-step, 9-channel inertial windows with a 1D CNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SIGNALS = [f"{signal}_{axis}" for signal in ("body_acc", "body_gyro", "total_acc")
           for axis in ("x", "y", "z")]


def load_signals(dataset_dir: Path, split: str) -> np.ndarray:
    signal_dir = dataset_dir / split / "Inertial Signals"
    arrays = [np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32) for name in SIGNALS]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or next(iter(shapes))[1] != 128:
        raise ValueError(f"Unexpected inertial signal shapes for {split}: {shapes}")
    return np.stack(arrays, axis=-1)  # samples x 128 time steps x 9 channels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "time_series_cnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    tf.keras.utils.set_random_seed(42)
    x_train = load_signals(args.data_dir, "train")
    x_test = load_signals(args.data_dir, "test")
    y_train = np.loadtxt(args.data_dir / "train" / "y_train.txt", dtype=np.int64) - 1
    y_test = np.loadtxt(args.data_dir / "test" / "y_test.txt", dtype=np.int64)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 9)),
        tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(), tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dense(6, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_split=0.15, epochs=args.epochs,
              batch_size=args.batch_size, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                                          restore_best_weights=True)])
    predicted = np.argmax(model.predict(x_test, verbose=0), axis=1) + 1
    labels = np.arange(1, 7)
    names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    report = classification_report(y_test, predicted, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "time_series_cnn.keras")
    np.savetxt(args.output_dir / "confusion_matrix.csv",
               confusion_matrix(y_test, predicted, labels=labels), fmt="%d", delimiter=",")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "accuracy": accuracy_score(y_test, predicted), "classification_report": report,
        "input_shape": [128, 9], "channels": SIGNALS,
    }, indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy_score(y_test, predicted):.4f}")
    print(classification_report(y_test, predicted, labels=labels, target_names=names, zero_division=0))
    print(f"Model and evaluation saved in {args.output_dir}")


if __name__ == "__main__":
    main()
