"""Train a dense neural network on the selected UCI HAR feature subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "selected_features.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"{args.input} not found; run 01_feature_selection.py first")

    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    tf.keras.utils.set_random_seed(42)
    data = np.load(args.input)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(data["X_train"].shape[1],)),
        tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.35), tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25), tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(data["X_train"], data["y_train"] - 1, validation_split=0.15,
              epochs=args.epochs, batch_size=args.batch_size, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                                          restore_best_weights=True)])
    predicted = np.argmax(model.predict(data["X_test"], verbose=0), axis=1) + 1
    truth = data["y_test"]
    labels = np.arange(1, 7)
    names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    report = classification_report(truth, predicted, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "dnn.keras")
    np.savetxt(args.output_dir / "confusion_matrix.csv",
               confusion_matrix(truth, predicted, labels=labels), fmt="%d", delimiter=",")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "accuracy": accuracy_score(truth, predicted), "classification_report": report,
        "n_features": int(data["X_train"].shape[1]),
    }, indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy_score(truth, predicted):.4f}")
    print(classification_report(truth, predicted, labels=labels, target_names=names, zero_division=0))
    print(f"Model and evaluation saved in {args.output_dir}")


if __name__ == "__main__":
    main()
