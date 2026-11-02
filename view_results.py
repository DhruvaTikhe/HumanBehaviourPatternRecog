"""Display feature-selection details and each model's test results in order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]
MODELS = [
    ("Random Forest", "random_forest"),
    ("DNN", "dnn"),
    ("Time Series CNN", "time_series_cnn"),
]


def print_confusion_matrix(path: Path) -> None:
    if not path.is_file():
        print("Confusion matrix not found. Run this model first.")
        return
    matrix = np.loadtxt(path, delimiter=",", dtype=int, ndmin=2)
    if matrix.shape != (len(ACTIVITIES), len(ACTIVITIES)):
        print(f"Unexpected confusion matrix dimensions: {matrix.shape}")
        return

    row_label_width = max(len("Actual \\ Predicted"), max(map(len, ACTIVITIES)))
    col_width = max(9, max(map(len, ACTIVITIES)))
    print("Actual \\ Predicted".ljust(row_label_width) + " | " + " | ".join(
        name.rjust(col_width) for name in ACTIVITIES
    ))
    print("-" * (row_label_width + 3 + len(ACTIVITIES) * col_width + 3 * (len(ACTIVITIES) - 1)))
    for name, row in zip(ACTIVITIES, matrix):
        print(name.ljust(row_label_width) + " | " + " | ".join(
            str(int(value)).rjust(col_width) for value in row
        ))
    print("Rows = actual class; columns = predicted class.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "selected_features.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    print("=" * 80)
    print("FEATURE SELECTION")
    print("=" * 80)
    if not args.input.is_file():
        print(f"Feature-selection archive not found: {args.input}")
        print("Run 01_feature_selection.py first.")
    else:
        with np.load(args.input, allow_pickle=False) as data:
            if "selected_names" not in data.files:
                print("No selected_names array in the archive.")
            else:
                names = [str(name) for name in data["selected_names"]]
                print(f"Selected feature count: {len(names)}")
                print("Selected feature names (comma-separated):")
                print(", ".join(names))

    for title, folder in MODELS:
        model_dir = args.output_dir / folder
        print("\n" + "=" * 80)
        print(title.upper())
        print("=" * 80)
        metrics_path = model_dir / "metrics.json"
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8") as file:
                metrics = json.load(file)
            print(f"Test accuracy: {metrics.get('accuracy', 'not recorded')}")
        else:
            print("Metrics not found. Run the model script first.")
        print("Confusion matrix (actual rows x predicted columns):")
        print_confusion_matrix(model_dir / "confusion_matrix.csv")


if __name__ == "__main__":
    main()
