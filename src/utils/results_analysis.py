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


def calculate_confidence_interval_95(acc: float, n_samples: int) -> float:
    """Calculates 95% confidence intervals."""
    return 1.96 * np.sqrt((acc * (1 - acc)) / n_samples)


def calculate_brier_scores(
    y_true: np.ndarray, probs: np.ndarray, class_cols: list
) -> np.ndarray:
    """Calculates the multi-class Brier Score per sample (lower is better)."""
    class_cols = [str(col).strip() for col in class_cols]
    col_to_idx = {cls_name: i for i, cls_name in enumerate(class_cols)}
    true_indices = np.array([col_to_idx[cls_name] for cls_name in y_true])

    expected_values = np.zeros_like(probs)
    expected_values[np.arange(len(true_indices)), true_indices] = 1

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
    true_indices = np.array([col_to_idx[lbl] for lbl in y_true])

    accuracies = predictions == true_indices

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # filter for samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # weighted absolute difference
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

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
            name = result_csv.stem.replace("_", ResultsAnalyser.model_name_seperator)
            self.model_to_df[name] = df

            class_cols = df.columns.drop(metadata_cols, errors="ignore").tolist()
            probs = df[class_cols].apply(pd.to_numeric).to_numpy()
            y_true = df["label"].to_numpy()
            y_pred = df[class_cols].idxmax(axis=1).to_numpy()

            # Assuming y_true is a 1D array of true labels and probs is the 2D probability matrix
            acc_top1 = top_k_accuracy_score(y_true, probs, k=1, labels=class_cols)
            acc_top5 = top_k_accuracy_score(y_true, probs, k=5, labels=class_cols)

            macro_class_acc = recall_score(y_true, y_pred, average="macro")

            model_brier_scores = calculate_brier_scores(y_true, probs, class_cols)
            model_ece = calculate_ece(y_true, probs, class_cols)

            self.binary_accuracies[name] = (y_pred == y_true).astype(int).tolist()
            self.brier_scores[name] = model_brier_scores.tolist()
            self.ece_scores[name] = model_ece

            self.results_table.append(
                {
                    "model": name,
                    "Class Accuracies": macro_class_acc * 100,
                    "Top-1": acc_top1 * 100,
                    "Top-5": acc_top5 * 100,
                    "Top-1 Error": calculate_confidence_interval_95(acc_top1, len(df))
                    * 100,
                    "Top-5 Error": calculate_confidence_interval_95(acc_top5, len(df))
                    * 100,
                    "Brier Score": model_brier_scores.mean(),
                    "ECE": model_ece * 100,
                }
            )

        self.accuracies = pd.DataFrame(self.results_table).sort_values(
            by=["Top-1", "Top-5"], ascending=False
        )

    def print_results_table(self):
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
            label="Top-5 Accuracy",
            color=color_top5,
        )

        # plot top-1
        top1_bars = ax.barh(
            y_pos + height / 2,
            results["Top-1"],
            height,
            xerr=results["Top-1 Error"],
            capsize=5,
            label="Top-1 Accuracy",
            color=color_top1,
        )

        ax.bar_label(top1_bars, fmt=" %.1f%%")
        ax.bar_label(top5_bars, fmt=" %.1f%%")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(results["model"])
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Pipeline Configuration")
        ax.set_title("Action Recognition Performance with 95% Confidence Intervals")

        # layout
        all_vals = np.concatenate(
            [
                results["Top-1"] - results["Top-1 Error"],
                results["Top-1"] + results["Top-1 Error"],
                results["Top-5"] - results["Top-5 Error"],
                results["Top-5"] + results["Top-5 Error"],
            ]
        )

        xmin = all_vals.min()
        xmax = all_vals.max()
        margin = (xmax - xmin) * 0.1

        ax.set_xlim(xmin - margin, xmax + margin)
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="grey")

        ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_accuracy_mcnemar_heatmap(self):
        """Pairwise McNemar's tests with Holm correction for hard Accuracy."""
        accuracies = {
            self.shorten_model_name(model): acc
            for model, acc in self.binary_accuracies.items()
        }
        binary_df = pd.DataFrame(accuracies)
        models = list(binary_df.columns)

        q_stat = cochrans_q(binary_df)
        print(f"Cochran's Q Test p-value: {q_stat.pvalue:.4f}")

        if q_stat.pvalue > 0.05:
            print("No statistically significant difference in accuracy found.")
            return

        pairs = list(itertools.combinations(models, 2))
        p_values_list = []

        for m1, m2 in pairs:
            table = pd.crosstab(binary_df[m1], binary_df[m2])
            table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            res = mcnemar(table, exact=False, correction=True)
            p_values_list.append(res.pvalue)

        _, p_adj, _, _ = multipletests(p_values_list, method="holm")

        p_mat = pd.DataFrame(
            np.ones((len(models), len(models))), index=models, columns=models
        )
        for (m1, m2), p in zip(pairs, p_adj):
            p_mat.loc[m1, m2] = p_mat.loc[m2, m1] = p

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(p_mat, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            p_mat < 0.05,
            annot=p_mat,
            mask=mask,
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.title(
            "Pairwise McNemar's Tests with Holm Correction\n(Blue indicates significant difference, p < 0.05)",
            pad=20,
        )
        plt.tight_layout()
        plt.show()

    def plot_calibration_wilcoxon_heatmap(self):
        """Pairwise Wilcoxon Signed-Rank tests with Holm correction for Brier Scores."""
        scores = {
            self.shorten_model_name(model): scores
            for model, scores in self.brier_scores.items()
        }
        brier_df = pd.DataFrame(scores)
        models = list(brier_df.columns)

        stat, p_friedman = st.friedmanchisquare(*[brier_df[col] for col in models])
        print(f"Friedman Test p-value: {p_friedman:.4f}")

        if p_friedman > 0.05:
            print("No statistically significant differences in calibration found.")
            return

        melted_df = brier_df.melt(var_name="model", value_name="score")
        p_mat = sp.posthoc_wilcoxon(
            melted_df, val_col="score", group_col="model", p_adjust="holm"
        )

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(p_mat, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            (p_mat < 0.05).astype(int),
            annot=p_mat,
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
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.title(
            "Pairwise Wilcoxon Tests for Calibration (Brier Score)\n(Green indicates significant difference, p < 0.05)",
            pad=20,
        )
        plt.tight_layout()
        plt.show()

    def plot_confused_classes_table(
        self, model_name: str, num_true_classes: int = 10, num_confused_with: int = 3
    ):
        """
        Plots a table showing classes with the most number of errors, their individual accuracy,
        and the specific classes they were most often confused with.
        """
        if model_name not in self.model_to_df:
            raise ValueError(f"Model {model_name} not found.")

        df = self.model_to_df[model_name]
        class_cols = df.columns.drop(metadata_cols, errors="ignore").tolist()

        y_true = df["label"]
        y_pred = df[class_cols].idxmax(axis=1)

        recalls = recall_score(y_true, y_pred, average=None, labels=class_cols)
        class_acc_dict = dict(zip(class_cols, recalls))

        errors_mask = y_pred != y_true

        per_class_recall = recall_score(y_true, y_pred, average=None, labels=class_cols)

        recall_df = pd.DataFrame(
            {"class": class_cols, "recall": per_class_recall}
        ).sort_values("recall")

        most_confused_classes = recall_df.head(num_true_classes)["class"].tolist()

        table_data = []
        for true_cls in most_confused_classes:
            # get samples for this true class that were wrong
            cls_errors = y_pred[(y_true == true_cls) & errors_mask]
            total_samples_in_cls = len(y_true[y_true == true_cls])

            confusion_counts = cls_errors.value_counts().head(num_confused_with)

            row = [true_cls, f"{class_acc_dict[true_cls] * 100:.1f}%"]

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

        fig, ax = plt.subplots()
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
            f"Top {num_true_classes} Problematic Classes\n(Model: {model_name})", pad=50
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

            class_cols = [c for c in df.columns if c not in metadata_cols]

            probs = df[class_cols].to_numpy()
            y_true = df["label"].to_numpy()

            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            col_to_idx = {cls: i for i, cls in enumerate(class_cols)}
            true_indices = np.array([col_to_idx[lbl] for lbl in y_true])
            accuracies = predictions == true_indices

            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_accs = []
            bin_confs = []

            for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                if np.mean(in_bin) > 0:
                    bin_accs.append(np.mean(accuracies[in_bin]))
                    bin_confs.append(np.mean(confidences[in_bin]))

            plt.plot(
                bin_confs,
                bin_accs,
                marker="o",
                label=f"{name} (ECE: {self.ece_scores[name] * 100:.1f}%)",
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
