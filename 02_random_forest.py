"""Train and evaluate Random Forest on the selected UCI HAR features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "selected_features.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "random_forest")
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"{args.input} not found; run 01_feature_selection.py first")

    data = np.load(args.input)
    model = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(data["X_train"], data["y_train"])
    predicted = model.predict(data["X_test"])
    labels = np.arange(1, 7)
    names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    report = classification_report(data["y_test"], predicted, labels=labels,
                                   target_names=names, output_dict=True, zero_division=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "confusion_matrix.csv",
               confusion_matrix(data["y_test"], predicted, labels=labels), fmt="%d", delimiter=",")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "accuracy": accuracy_score(data["y_test"], predicted), "classification_report": report,
        "n_features": int(data["X_train"].shape[1]), "n_estimators": args.trees,
    }, indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy_score(data['y_test'], predicted):.4f}")
    print(classification_report(data["y_test"], predicted, labels=labels, target_names=names, zero_division=0))
    print(f"Metrics and confusion matrix saved in {args.output_dir}")


if __name__ == "__main__":
    main()
