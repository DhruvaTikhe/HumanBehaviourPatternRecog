"""Create four plots summarizing the three UCI HAR models.

Run after feature selection and all model scripts:
    python visualize_results.py
    python visualize_results.py --no-show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_classif
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
ACTIVITY_NAMES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
REPORT_CLASS_KEYS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
MODEL_DIRS = ["random_forest", "dnn", "time_series_cnn"]
MODEL_NAMES = ["Random Forest", "DNN", "Time-Series CNN"]
COLORS = ["#3468a5", "#e59b3a", "#3a9d78"]


def load_model_results(output_dir: Path) -> tuple[list[dict], list[np.ndarray]]:
    reports: list[dict] = []
    matrices: list[np.ndarray] = []
    for directory in MODEL_DIRS:
        model_dir = output_dir / directory
        metrics_path = model_dir / "metrics.json"
        matrix_path = model_dir / "confusion_matrix.csv"
        if not metrics_path.is_file() or not matrix_path.is_file():
            raise FileNotFoundError(
                f"Missing results for {directory}. Run its model script before visualizing."
            )
        reports.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        matrix = np.loadtxt(matrix_path, delimiter=",", dtype=int, ndmin=2)
        if matrix.shape != (6, 6):
            raise ValueError(f"Expected a 6x6 confusion matrix in {matrix_path}, got {matrix.shape}")
        matrices.append(matrix)
    return reports, matrices


def save_figure(fig: plt.Figure, output_dir: Path, name: str, show: bool,
                figures: list[plt.Figure]) -> None:
    path = output_dir / f"graphtype_{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved: {path.resolve()}")
    if show:
        figures.append(fig)
    else:
        plt.close(fig)


def plot_pearson_heatmap(archive_path: Path, output_dir: Path, top_features: int,
                         show: bool, figures: list[plt.Figure]) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Feature archive not found: {archive_path}. Run 01_feature_selection.py first.")
    with np.load(archive_path, allow_pickle=False) as data:
        train_x = data["X_train"]
        train_y = data["y_train"]
        names = data["selected_names"].astype(str)

    k = min(top_features, train_x.shape[1])
    scores, _ = f_classif(train_x, train_y)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    chosen = np.argsort(scores)[-k:][::-1]
    selected_x = train_x[:, chosen]
    selected_names = names[chosen]
    correlation = np.nan_to_num(np.corrcoef(selected_x, rowvar=False), nan=0.0)

    fig, ax = plt.subplots(figsize=(14, 12), layout="constrained")
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"Pearson correlation: top {k} selected features", fontsize=15, weight="bold")
    ax.set_xticks(range(k), selected_names, rotation=90, fontsize=7)
    ax.set_yticks(range(k), selected_names, fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation coefficient (r)", shrink=0.8)
    save_figure(fig, output_dir, "pearson_correlation", show, figures)


def plot_horizontal_scores(reports: list[dict], output_dir: Path, show: bool,
                           figures: list[plt.Figure]) -> None:
    accuracy = np.array([r["accuracy"] for r in reports]) * 100
    macro_f1 = np.array([r["classification_report"]["macro avg"]["f1-score"] for r in reports]) * 100
    y = np.arange(len(MODEL_NAMES))
    height = 0.34
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    acc_bars = ax.barh(y - height / 2, accuracy, height, label="Accuracy", color="#3468a5")
    f1_bars = ax.barh(y + height / 2, macro_f1, height, label="Macro F1", color="#e59b3a")
    ax.set_yticks(y, MODEL_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Held-out test score (%)")
    ax.set_title("Model comparison", fontsize=15, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.bar_label(acc_bars, fmt="%.1f%%", padding=4)
    ax.bar_label(f1_bars, fmt="%.1f%%", padding=4)
    save_figure(fig, output_dir, "model_comparison", show, figures)


def plot_classwise_f1(reports: list[dict], output_dir: Path, show: bool,
                      figures: list[plt.Figure]) -> None:
    values = np.array([
        [report["classification_report"][key]["f1-score"] * 100
         for key in REPORT_CLASS_KEYS]
        for report in reports
    ])
    x = np.arange(len(ACTIVITY_NAMES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    for index, (name, color) in enumerate(zip(MODEL_NAMES, COLORS)):
        bars = ax.bar(x + (index - 1) * width, values[index], width,
                      label=name, color=color)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x, ACTIVITY_NAMES)
    ax.set_ylim(0, 112)
    ax.set_ylabel("F1 score (%)")
    ax.set_title("Per-activity F1 score", fontsize=15, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center")
    save_figure(fig, output_dir, "classwise_f1", show, figures)


def plot_confusion_matrices(matrices: list[np.ndarray], output_dir: Path, show: bool,
                            figures: list[plt.Figure]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    for ax, name, matrix in zip(axes, MODEL_NAMES, matrices):
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
                               where=row_totals != 0)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(name, fontsize=12, weight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(6), ACTIVITY_NAMES, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
        ax.set_yticks(range(6), ACTIVITY_NAMES, fontsize=8)
        for row in range(6):
            for col in range(6):
                pct = normalized[row, col]
                ax.text(col, row, f"{pct:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if pct >= 0.55 else "#17202a")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                     format=FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle("Row-normalized confusion matrices", fontsize=15, weight="bold")
    save_figure(fig, output_dir, "normalized_confusion_matrices", show, figures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs" / "selected_features.npz")
    parser.add_argument("--top-features", type=int, default=20,
                        help="number of top selected features in the correlation heat map")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = parser.parse_args()
    if args.top_features < 2:
        parser.error("--top-features must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, matrices = load_model_results(args.output_dir)
    open_figures: list[plt.Figure] = []
    show = not args.no_show

    plot_pearson_heatmap(args.archive, args.output_dir, args.top_features, show, open_figures)
    plot_horizontal_scores(reports, args.output_dir, show, open_figures)
    plot_classwise_f1(reports, args.output_dir, show, open_figures)
    plot_confusion_matrices(matrices, args.output_dir, show, open_figures)

    if show:
        plt.show()


if __name__ == "__main__":
    main()