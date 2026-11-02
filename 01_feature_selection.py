"""Select the K most informative precomputed UCI HAR features.

Run from any directory with: python 01_feature_selection.py --k 100
The selector is fitted on training data only; the official test set is untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


ROOT = Path(__file__).resolve().parent


def read_labels(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=100, help="number of features to retain (default: 100)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    train_x = np.loadtxt(args.data_dir / "train" / "X_train.txt")
    test_x = np.loadtxt(args.data_dir / "test" / "X_test.txt")
    train_y = read_labels(args.data_dir / "train" / "y_train.txt")
    test_y = read_labels(args.data_dir / "test" / "y_test.txt")
    feature_names = np.array([line.split(maxsplit=1)[1] for line in
                              (args.data_dir / "features.txt").read_text(encoding="utf-8").splitlines()])

    if not (1 <= args.k <= train_x.shape[1]):
        parser.error(f"--k must be between 1 and {train_x.shape[1]}")
    if train_x.shape[1] != len(feature_names) or test_x.shape[1] != len(feature_names):
        raise ValueError("Feature matrix width does not match features.txt")

    selector = SelectKBest(score_func=f_classif, k=args.k).fit(train_x, train_y)
    indices = selector.get_support(indices=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "selected_features.npz"
    np.savez_compressed(
        destination,
        X_train=selector.transform(train_x), X_test=selector.transform(test_x),
        y_train=train_y, y_test=test_y,
        selected_indices=indices, selected_names=feature_names[indices],
    )
    print(f"Selected {args.k}/{train_x.shape[1]} features using training data only.")
    print(f"Saved train/test subsets and labels to {destination}")
    print("Selected features:", ", ".join(feature_names[indices]))


if __name__ == "__main__":
    main()
