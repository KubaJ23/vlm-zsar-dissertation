from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from pandas import DataFrame

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


def calculate_top_n_accuracy(df: DataFrame, n: int) -> float:
    """
    Calculates the top-n accuracy.
    """

    class_cols = df.columns.drop(metadata_cols, errors="ignore")

    # create probability matrix, shape (num_videos, num_classes)
    probs = df[class_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

    if n >= probs.shape[1]:
        return 1.0

    # Get top n indices
    top_n_indices = np.argpartition(probs, -n, axis=1)[:, -n:]

    # Map class names to indices
    col_to_idx = {col: i for i, col in enumerate(class_cols)}
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

        self.data = []
        self.result_dfs = {}

        for result_csv in results_dir.glob("*.csv"):
            df = pd.read_csv(result_csv)
            name = result_csv.stem.replace("_", ", ")

            self.result_dfs[name] = df

            acc_top1 = calculate_top_n_accuracy(df, 1)
            acc_top5 = calculate_top_n_accuracy(df, 5)
            n_samples = len(df)

            # Calculate 95% confidence intervals
            err_top1 = 1.96 * np.sqrt((acc_top1 * (1 - acc_top1)) / n_samples)
            err_top5 = 1.96 * np.sqrt((acc_top5 * (1 - acc_top5)) / n_samples)

            self.data.append(
                {
                    "model": name,
                    "Top-1": acc_top1 * 100,
                    "Top-5": acc_top5 * 100,
                    "Top-1 Error": err_top1 * 100,
                    "Top-5 Error": err_top5 * 100,
                }
            )

        self.accuracies = pd.DataFrame(self.data).sort_values(
            by=["Top-1", "Top-5"], ascending=False
        )

    def print_results_table(self):
        print(self.accuracies.to_string(index=False, float_format="{:.2f}".format))

    def show_accuracy_comparison_plot(self):
        """Plots a horizontal bar chart with the different models on the y axis and their accuracies on the x axis"""

        results = self.accuracies.sort_values(by=["Top-1", "Top-5"], ascending=True)

        num_models = len(results)
        fig, ax = plt.subplots(figsize=(16, max(6, num_models * 0.9)))

        y_pos = np.arange(num_models)
        height = 0.35

        color_top1 = "#60A5FA"
        color_top5 = "#CBD5E1"

        # plot top-5
        top5_bars = ax.barh(
            y_pos - height / 2,
            results["Top-5"],
            height,
            xerr=results["Top-5 Error"],
            capsize=5,
            error_kw={"ecolor": "black", "elinewidth": 1.5, "capthick": 1.5},
            label="Top-5 Accuracy",
            color=color_top5,
            edgecolor="#94A3B8",
        )

        # plot top-1
        top1_bars = ax.barh(
            y_pos + height / 2,
            results["Top-1"],
            height,
            xerr=results["Top-1 Error"],
            capsize=5,
            error_kw={"ecolor": "black", "elinewidth": 1.5, "capthick": 1.5},
            label="Top-1 Accuracy",
            color=color_top1,
            edgecolor="#2563EB",
        )

        ax.bar_label(
            top1_bars, fmt=" %.1f%%", padding=8, fontweight="bold", fontsize=11
        )
        ax.bar_label(top5_bars, fmt=" %.1f%%", padding=8, fontsize=10, color="dimgrey")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(results["model"], fontsize=11, fontweight="500")
        ax.set_xlabel("Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pipeline Configuration", fontsize=12, fontweight="bold")
        ax.set_title(
            "Action Recognition Performance with 95% Confidence Intervals",
            fontsize=14,
            fontweight="bold",
            pad=40,
        )

        # layout
        ax.margins(x=0.15)
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="grey")
        ax.spines[["top", "right"]].set_visible(False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=2,
            frameon=False,
            fontsize=11,
        )

        plt.tight_layout()
        plt.show()

    def plot_cd_diagram(self):
        """
        Performs a Pairwise Wilcoxon Signed-Rank Tests with Holm correction.

        Plots a critical difference diagram for all models using top 1 accuracy.
        Models connected by a horizontal line are not significantly different at p < 0.05.
        Better models have lower rankes so are on the left.
        """

        results_dict = {}
        for name, df in self.result_dfs.items():
            class_cols = df.columns.drop(metadata_cols, errors="ignore")
            preds = df[class_cols].idxmax(axis=1)
            # Get vector of 1/0 for correct/incorrect predictions
            results_dict[name] = (preds == df["label"]).astype(int).values

        # df with each row for a video and each column for a pipeline
        test_results_df = pd.DataFrame(results_dict)

        # Average ranks across videos for each model
        ranks = test_results_df.rank(axis=1, ascending=False).mean()

        p_values = sp.posthoc_wilcoxon(
            test_results_df.melt(var_name="model", value_name="score"),
            val_col="score",
            group_col="model",
            p_adjust="holm",
        )

        plt.figure(figsize=(12, 4), dpi=100)
        plt.title(
            "Critical Difference Diagram (Top-1 Accuracy)\nWilcoxon-Holm (p < 0.05)",
            pad=20,
            fontweight="bold",
        )

        sp.critical_difference_diagram(ranks, p_values, ax=None)

        plt.show()
