import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as st
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, recall_score, top_k_accuracy_score
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

font_size = 15
plt.rcParams.update(
    {
        "font.size": font_size,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "font.family": "serif",
    }
)

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
    "video_id",
    "fo_id",
]


def calculate_brier_scores(
    y_true: np.ndarray, probs: np.ndarray, class_cols: list
) -> np.ndarray:
    """calculates the brier Score per video"""
    col_to_idx = {cls_name: i for i, cls_name in enumerate(class_cols)}
    true_indices = np.array([col_to_idx[cls_name] for cls_name in y_true])

    # create matrix of 0s and 1 for the expected values for each video and class
    expected_values = np.zeros_like(probs)
    expected_values[np.arange(len(true_indices)), true_indices] = 1

    # get brier score by calculating difference between expected and predicted probs
    return np.sum((probs - expected_values) ** 2, axis=1)


def calculate_ece(
    y_true: np.ndarray, probs: np.ndarray, class_cols: list, n_bins=10
) -> float:
    """
    Calculates Expected Calibration Error (ECE).
    Lower is better. Perfect calibration = 0.0.
    """
    # Get confidence (max prob) and prediction (argmax)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    # Map true labels to indices
    col_to_idx = {cls: i for i, cls in enumerate(class_cols)}
    true_idxs = np.array([col_to_idx[label] for label in y_true])

    accuracies = predictions == true_idxs

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # filter for samples in this confidence bin
        in_bin = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        fraction_inside_bin = np.mean(in_bin)

        if fraction_inside_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # weighted absolute difference
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * fraction_inside_bin

    return ece


class ResultsAnalyser:
    """
    Analyses the result files of experiments, providing methods for easy and reusable analysis outputs.
    Expects a directory of CSV files (the result files) with columns for each class, and probabilities between each video and each class.

    This class helps to avoid repeating the same code for the results analysis in different notebooks.
    """

    model_name_seperator = ", "

    def __init__(self, results_dir: Path):
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        self.results_table = []
        self.model_to_df = {}

        self.binary_accuracies = {}
        self.brier_scores = {}
        self.ece_scores = {}

        for result_csv in results_dir.glob("*.csv"):
            df = pd.read_csv(result_csv)
            pipeline_name = result_csv.stem.replace(
                "_", ResultsAnalyser.model_name_seperator
            )
            self.model_to_df[pipeline_name] = df

            class_cols = df.columns.drop(metadata_cols, errors="ignore").tolist()
            probs = df[class_cols].apply(pd.to_numeric).to_numpy()
            y_true = df["label"].to_numpy()
            y_pred = df[class_cols].idxmax(axis=1).to_numpy()

            # Assuming y_true is a 1D array of true labels and probs is the 2D probability matrix
            acc_top1 = top_k_accuracy_score(y_true, probs, k=1, labels=class_cols)
            acc_top5 = top_k_accuracy_score(y_true, probs, k=5, labels=class_cols)

            per_class_acc = recall_score(y_true, y_pred, average="macro")

            model_brier_scores = calculate_brier_scores(y_true, probs, class_cols)
            model_ece = calculate_ece(y_true, probs, class_cols)

            self.binary_accuracies[pipeline_name] = (
                (y_pred == y_true).astype(int).tolist()
            )
            self.brier_scores[pipeline_name] = model_brier_scores.tolist()
            self.ece_scores[pipeline_name] = model_ece

            self.results_table.append(
                {
                    "model": pipeline_name,
                    "Class Accuracies": per_class_acc * 100,
                    "Top-1": acc_top1 * 100,
                    "Top-5": acc_top5 * 100,
                    "Brier Score": model_brier_scores.mean(),
                    "ECE": model_ece * 100,
                }
            )

        self.accuracies = pd.DataFrame(self.results_table).sort_values(
            by=["Top-1", "Top-5"], ascending=False
        )

    def print_results_table(self):
        # print with a general format for printing floats, but brier scores and ECE need more decimal places because the results are so similar between pipelines.
        print(
            self.accuracies.to_string(
                index=False,
                float_format="{:.1f}".format,
                formatters={
                    "Brier Score": "{:.6f}".format,
                    "ECE": "{:.2f}%".format,
                },
            )
        )

    def show_accuracy_comparison_plot(self):
        """Plots a horizontal bar chart with the different models on the y axis and their accuracies on the x axis"""

        results = self.accuracies.sort_values(by="Top-1", ascending=True).copy()
        # maake hte pipeline name labels shorter so graph can be bigger and clearer
        results["model"] = results["model"].apply(self.shorten_model_name)

        num_models = len(results)
        fig, ax = plt.subplots(figsize=(16, max(6, num_models)))

        y_pos = np.arange(num_models)
        height = 0.35

        color_top1 = "#60A5FA"
        color_top5 = "#CBD5E1"

        # plot top-5
        top5_bars = ax.barh(
            y_pos - height / 2,
            results["Top-5"],
            height,
            label="Top-5 Accuracy",
            color=color_top5,
        )

        # plot top-1
        top1_bars = ax.barh(
            y_pos + height / 2,
            results["Top-1"],
            height,
            label="Top-1 Accuracy",
            color=color_top1,
        )

        # add the labels per bar so specific accuracy can be read
        ax.bar_label(top1_bars, fmt=" %.1f%%")
        ax.bar_label(top5_bars, fmt=" %.1f%%")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(results["model"])
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Pipeline Configuration")
        ax.set_title("Action Recognition Accuracy")

        # set correct limits on x axis so bar chart is sized correctly and not small
        xmin = min(results["Top-1"].min(), results["Top-5"].min())
        xmax = max(results["Top-1"].max(), results["Top-5"].max())

        xpadding = (xmax - xmin) * 0.1

        ax.set_xlim(xmin - xpadding, xmax + xpadding)
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="grey")

        plt.show()

    def plot_accuracy_mcnemar_heatmap(self):
        """Pairwise McNemar's tests with Holm correction for hard Accuracy."""
        binary_df = pd.DataFrame(
            {
                self.shorten_model_name(model): acc
                for model, acc in self.binary_accuracies.items()
            }
        )
        models = list(binary_df.columns)

        if cochrans_q(binary_df).pvalue > 0.05:
            print("No statistically significant difference in accuracy found.")
            return

        pairs = list(itertools.combinations(models, 2))
        p_values_list = []

        for m1, m2 in pairs:
            # make a table of where the 2 models agree, disagree and how often one model is correct while the other isnt
            table = pd.crosstab(binary_df[m1], binary_df[m2])
            # ensure the table is 2x2
            table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            mcnemar_res = mcnemar(table, exact=False, correction=True)
            p_values_list.append(mcnemar_res.pvalue)

        # correct the p values for multiple comparisons
        _, corrected_ps, _, _ = multipletests(p_values_list, method="holm")

        p_table = pd.DataFrame(
            np.ones((len(models), len(models))), index=models, columns=models
        )
        for (m1, m2), p in zip(pairs, corrected_ps):
            p_table.loc[m1, m2] = p_table.loc[m2, m1] = p

        # Create a mask for the upper triangle
        triangle_mask = np.triu(np.ones_like(p_table, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            p_table < 0.05,
            annot=p_table,
            mask=triangle_mask,
            cmap="Blues",
            cbar=False,
            ax=ax,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        plt.title(
            "Pairwise McNemar's Tests with Holm Correction\n(Blue indicates significant difference, p < 0.05)",
            pad=20,
        )
        plt.show()

    def plot_calibration_wilcoxon_heatmap(self):
        """Pairwise Wilcoxon Signed-Rank tests with Holm correction for Brier Scores."""
        brier_df = pd.DataFrame(
            {
                self.shorten_model_name(model): scores
                for model, scores in self.brier_scores.items()
            }
        )
        models = list(brier_df.columns)

        _, p_friedman = st.friedmanchisquare(*[brier_df[col] for col in models])
        print(f"friedman test p-value: {p_friedman:.4f}")

        if p_friedman > 0.05:
            print("No statistically significant differences in calibration found.")
            return

        # change the brier_df shape to have a column for model and score
        brier_df = brier_df.melt(var_name="model", value_name="score")
        p_table = sp.posthoc_wilcoxon(
            brier_df, val_col="score", group_col="model", p_adjust="holm"
        )

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(p_table, dtype=bool))

        _, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            (p_table < 0.05).astype(int),
            annot=p_table,
            mask=mask,
            cmap="Greens",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        plt.title(
            "Pairwise Wilcoxon Tests for Calibration (Brier Score)\n(Green indicates significant difference, p < 0.05)",
            pad=20,
        )
        plt.show()

    def plot_confused_classes_table(
        self, model_name: str, num_true_classes: int = 10, num_confused_with: int = 3
    ):
        """
        Plots a table showing classes with the worst accuracy, their individual accuracy,
        and the specific classes they were most often confused with.
        """
        if model_name not in self.model_to_df:
            raise ValueError(f"Model {model_name} not found.")

        df = self.model_to_df[model_name]
        class_cols = df.columns.drop(metadata_cols, errors="ignore").tolist()

        y_true = df["label"]
        y_pred = df[class_cols].idxmax(axis=1)

        # get recall for each class in order of class_cols
        class_accs = recall_score(y_true, y_pred, average=None, labels=class_cols)

        errors_mask = y_pred != y_true

        recall_df = pd.DataFrame(
            {"class": class_cols, "recall": class_accs}
        ).sort_values("recall")

        most_confused_classes = recall_df.head(num_true_classes)["class"].tolist()

        table_data = []
        for true_cls in most_confused_classes:
            # get samples for this true class that were wrong
            cls_errors = y_pred[(y_true == true_cls) & errors_mask]
            total_samples_in_cls = len(y_true[y_true == true_cls])

            confusion_counts = cls_errors.value_counts().head(num_confused_with)

            row = [true_cls, f"{class_accs[class_cols.index(true_cls)] * 100:.1f}%"]

            for i in range(num_confused_with):
                if i < len(confusion_counts):
                    wrong_cls = confusion_counts.index[i]
                    count = confusion_counts.values[i]
                    # Percentage of total samples of the True Class mislabelled as this specific wrong class
                    wrong_percentage = (count / total_samples_in_cls) * 100
                    row.append(f"{wrong_cls}\n({wrong_percentage:.1f}%)")
                else:
                    row.append("-")
            table_data.append(row)

        columns = ["True Class", "Class Accuracy\n(Recall)"] + [
            f"Mislabelled As\n(Rank {i + 1})" for i in range(num_confused_with)
        ]

        _, ax = plt.subplots()
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc="center",
            loc="center",
            colColours=["#E0E0E0"] * len(columns),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(2.2, 2.6)

        plt.title(
            f"Top {num_true_classes} Problematic Classes (Model: {self.shorten_model_name(model_name)})",
            pad=50,
        )
        plt.show()

    def plot_reliability_diagram(self, models: list[str], n_bins=10):
        """
        Plots the Calibration Curve for selected models. Ideally, lines should lie on the diagonal.
        """
        plt.figure(figsize=(10, 8))

        # plot perfect calibration line
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )

        for name, df in self.model_to_df.items():
            if name not in models:
                continue

            class_cols = df.columns.drop(metadata_cols, errors="ignore").tolist()

            probs = df[class_cols].to_numpy()
            y_true = df["label"].to_numpy()

            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            col_to_idx = {cls: i for i, cls in enumerate(class_cols)}
            true_indices = np.array([col_to_idx[label] for label in y_true])
            accuracies = predictions == true_indices

            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_accs = []
            bin_confs = []

            for i in range(n_bins):
                in_bin = (confidences > bin_boundaries[i]) & (
                    confidences <= bin_boundaries[i + 1]
                )
                if np.mean(in_bin) > 0:
                    bin_accs.append(np.mean(accuracies[in_bin]))
                    bin_confs.append(np.mean(confidences[in_bin]))

            plt.plot(
                bin_confs,
                bin_accs,
                marker="o",
                label=f"{self.shorten_model_name(name)} (ECE: {self.ece_scores[name] * 100:.1f}%)",
            )

        plt.xlabel("Predicted Probability")
        plt.ylabel("Accuracy")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def shorten_model_name(self, model_name: str) -> str:
        (sampler, aggregator, prompter) = model_name.split(
            ResultsAnalyser.model_name_seperator
        )
        sampler = "MGS" if "mgsampler" in sampler.lower() else "Uni"
        aggregator = "QS" if "queryscoring" in aggregator.lower() else "MP"
        prompter = "MPVR" if "mpvr" in prompter.lower() else "Temp"

        return f"{sampler}{ResultsAnalyser.model_name_seperator}{aggregator}{ResultsAnalyser.model_name_seperator}{prompter}"
