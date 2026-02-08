from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_top_n_accuracy(df: DataFrame, n: int) -> float:
    """
    Calculates the top-n accuracy.
    """
    metadata_cols = [
        "clip_name",
        "clip_path",
        "label",
        "frame_count",
        "duration_sec",
        "fps",
        "width",
        "height",
        "resolution",
    ]

    class_cols = df.columns.drop(metadata_cols, errors="ignore")

    # create probability matrix, shape (num_videos, num_classes)
    probs = df[class_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

    if n >= probs.shape[1]:
        return 1.0

    # get top n indices for each row at once
    top_n_indices = np.argpartition(probs, -n, axis=1)[:, -n:]

    # create map of class name to column index
    col_to_idx = {col: i for i, col in enumerate(class_cols)}

    # convert label column to indices, shape (num_videos,)
    true_label_indices = df["label"].map(col_to_idx).fillna(-1).to_numpy()

    matches = top_n_indices == true_label_indices[:, None]

    return matches.any(axis=1).mean()


class ResultsAnalyser:
    """
    Analyses the result files of experiments, providing methods for easy and reusable analysis outputs.
    Expects a directory of CSV files (the result files) with columns for each class, and probabilities between each video and each class.

    This class helps to avoid repeating the same code for the results analysis in different notebooks.
    """

    def __init__(self, results_dir: Path):
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        self.results_dir = results_dir

        result_files: dict[str, pd.DataFrame] = {}

        for result_csv in results_dir.glob("*.csv"):
            combination_name = result_csv.stem
            df = pd.read_csv(result_csv)
            result_files[combination_name] = df

        self.data = [
            {
                "model": name.replace("_", ", "),
                "Top-1": calculate_top_n_accuracy(df, 1) * 100,
                "Top-5": calculate_top_n_accuracy(df, 5) * 100,
            }
            for name, df in result_files.items()
        ]

        self.accuracies = pd.DataFrame(self.data).sort_values(
            by=["Top-1", "Top-5"], ascending=False
        )

    def print_results_table(self):
        """Prints table of results, including model name and accuracies (sorted by accuracy)"""
        print(self.accuracies.to_string(index=False, float_format="{:.2f}".format))

    def show_accuracy_comparison_plot(self):
        """Plots a horizontal bar chart with the different models on the y axis and their accuracies on the x axis"""

        results = self.accuracies.sort_values(by=["Top-1", "Top-5"], ascending=True)

        # plot with dynamic height
        num_models = len(results)
        fig, ax = plt.subplots(figsize=(16, max(6, num_models * 0.8)))

        y_pos = np.arange(num_models)
        height = 0.35

        # Plot the double bars for both accuracies
        top5_bars = ax.barh(
            y_pos - height / 2,
            results["Top-5"],
            height,
            label="Top-5 Accuracy",
            color="#a8dadc",
            edgecolor="grey",
        )

        top1_bars = ax.barh(
            y_pos + height / 2,
            results["Top-1"],
            height,
            label="Top-1 Accuracy",
            color="#1d3557",
            edgecolor="black",
        )

        # add percentage labels
        ax.bar_label(
            top1_bars, fmt=" %.1f%%", padding=8, fontweight="bold", fontsize=10
        )
        ax.bar_label(top5_bars, fmt=" %.1f%%", padding=8, fontsize=9, color="dimgrey")

        # configure axes and titles
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results["model"], fontsize=10)
        ax.set_xlabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pipeline Configuration", fontsize=12, fontweight="bold")
        ax.set_title(
            "Action Recognition Performance Comparison",
            fontsize=14,
            fontweight="bold",
            pad=35,
        )

        # Add margins to prevent labels from being cut off
        ax.margins(x=0.1)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

        plt.tight_layout()
        plt.show()
