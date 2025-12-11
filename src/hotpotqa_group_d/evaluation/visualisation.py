import json

import matplotlib.pyplot as plt
import numpy as np

# Global changes for readability
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "font.family": "serif",
    }
)


def extract_results(label, file_path):
    """
    Extract the resulting metrecis em, f1, percision and recall. Label the results

    Args:
        label (str): Results label
        file_path (str): File path for results file

    Returns:
        results (dict): Dictionary with metrics and label
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        results = dict()

        # Extract metrics
        results["label"] = label
        results["em"] = data["em"]
        results["f1"] = data["f1"]
        results["prec"] = data["prec"]
        results["recall"] = data["recall"]

        return results


def plot_metrics(results, file_path):
    """
    Method to plot resulting metrics from different files into the same plot for comparison
    Saves plot as pdf

    Args:
        results (list[dict]): List of result metrics and their label
        file_path (str): File path for resulting pdf file
    """

    # Style
    plt.style.use("petroff10")

    # Create figure
    fig, x_axis = plt.subplots(figsize=(6, 5))

    labels = [result["label"] for result in results]
    metrics = ["em", "f1", "prec", "recall"]

    # config
    x_locs = np.arange(len(labels))
    bar_width = 0.1
    multiplier = 0

    for metric in metrics:

        # Get all values for metric
        values = [result[metric] for result in results]

        # Calculate offset
        offset = multiplier * bar_width

        rects = x_axis.bar(x_locs + offset, values, bar_width, label=metric.upper())
        multiplier += 1

    x_axis.set_ylabel("Score")
    x_axis.set_title("Model Results Comparison")

    # Center text under bar cluster
    x_axis.set_xticks(x_locs + bar_width * 1.5)
    x_axis.set_xticklabels(labels)

    x_axis.legend(loc="upper right", ncols=4)

    x_axis.set_ylim(0, 0.7)

    # Add grid for readability
    x_axis.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(file_path, format="pdf", bbox_inches="tight")


def spider_plot(results, file_path):
    """
    Method to plot resulting metrics from different files into a spider plot for comparison
    Saves plot as pdf

    Args:
        results (list[dict]): List of result metrics and their label
        file_path (str): File path for resulting pdf file
    """

    # Style
    plt.style.use("petroff10")

    metrics = ["em", "f1", "prec", "recall"]
    num_metrics = len(metrics)

    # Offset angles so they don't align with axis
    offset = np.pi / 4

    # Angles for graph
    angles = (np.linspace(0, 2 * np.pi, num_metrics, endpoint=False) + offset).tolist()
    angles += angles[:1]  # Add first element at the end to close loop

    # Create polar figure
    fig, x_axis = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for result in results:
        values = [result[metric] for metric in metrics]
        values += values[:1]  # Add first element at the end to close loop

        x_axis.plot(angles, values, linewidth=2, label=result["label"])

    # Exclude last entry cause it's dupelicated for loop purposes
    x_axis.set_xticks(angles[:-1])
    x_axis.set_xticklabels(metrics, fontsize=24)
    x_axis.set_rlabel_position(0)  # type: ignore

    plt.ylim(0, 0.7)
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1.4), fontsize=18)
    plt.title("Model Performance Comparison", fontsize=24)

    plt.savefig(file_path, format="pdf", bbox_inches="tight")
