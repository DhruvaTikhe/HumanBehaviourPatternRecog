"""Plot test-set scores and confusion matrices for the three HAR models.

Run after the model scripts, for example:
    python visualize_results.py
    python visualize_results.py --no-show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ACTIVITY_NAMES = [
    "Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying",
]
MODEL_DIRS = ["random_forest", "dnn", "time_series_cnn"]
MODEL_NAMES = ["Random Forest", "DNN", "Time-Series CNN"]


def load_results(output_dir: Path) -> tuple[list[float], list[float], list[np.ndarray]]:
    accuracies: list[float] = []
    macro_f1: list[float] = []
    matrices: list[np.ndarray] = []

    for directory in MODEL_DIRS:
        model_dir = output_dir / directory
        metrics_path = model_dir / "metrics.json"
        matrix_path = model_dir / "confusion_matrix.csv"
        if not metrics_path.is_file() or not matrix_path.is_file():
            raise FileNotFoundError(
                f"Missing results for {directory}. Run its model script before plotting."
            )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        accuracies.append(float(metrics["accuracy"]))
        macro_f1.append(float(metrics["classification_report"]["macro avg"]["f1-score"]))
        matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int, ndmin=2)
        if matrix.shape != (6, 6):
            raise ValueError(f"Expected a 6x6 confusion matrix in {matrix_path}, got {matrix.shape}")
        matrices.append(matrix)

    return accuracies, macro_f1, matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_comparison.png")
    parser.add_argument("--no-show", action="store_true", help="save the plot without opening a window")
    args = parser.parse_args()

    accuracies, macro_f1, matrices = load_results(args.output_dir)
    fig = plt.figure(figsize=(17, 12), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.5])
    score_ax = fig.add_subplot(grid[0, :])
    positions = np.arange(len(MODEL_NAMES))
    width = 0.34
    bars_acc = score_ax.bar(positions - width / 2, np.array(accuracies) * 100, width,
                            label="Accuracy", color="#3468a5")
    bars_f1 = score_ax.bar(positions + width / 2, np.array(macro_f1) * 100, width,
                           label="Macro F1", color="#e59b3a")
    score_ax.set_title("Held-out test performance", fontsize=15, weight="bold")
    score_ax.set_ylabel("Score (%)")
    score_ax.set_ylim(0, 108)
    score_ax.set_xticks(positions, MODEL_NAMES)
    score_ax.grid(axis="y", alpha=0.25)
    score_ax.legend(frameon=False, ncols=2, loc="upper right")
    score_ax.bar_label(bars_acc, fmt="%.1f%%", padding=3, fontsize=10)
    score_ax.bar_label(bars_f1, fmt="%.1f%%", padding=3, fontsize=10)

    for index, (name, matrix) in enumerate(zip(MODEL_NAMES, matrices)):
        ax = fig.add_subplot(grid[1, index])
        row_totals = matrix.sum(axis=1, keepdims=True)
        row_percent = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
                                 where=row_totals != 0)
        ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(name, fontsize=12, weight="bold")
        ax.set_xlabel("Predicted activity")
        ax.set_ylabel("Actual activity")
        ax.set_xticks(range(6), ACTIVITY_NAMES, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(6), ACTIVITY_NAMES)
        for row in range(6):
            for col in range(6):
                percent = row_percent[row, col]
                color = "white" if percent >= 0.55 else "#17202a"
                ax.text(col, row, f"{matrix[row, col]}\n{percent:.0%}",
                        ha="center", va="center", fontsize=8, color=color)

    fig.suptitle("UCI HAR: model comparison and confusion matrices", fontsize=17, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved visualization to {args.output.resolve()}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
