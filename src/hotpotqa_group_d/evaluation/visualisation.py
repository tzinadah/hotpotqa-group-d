import json


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
        results["precision"] = data["precision"]
        results["recall"] = data["recall"]

        return results


def plot_metrics(file_paths):
    """
    Method to plot resulting metrics from different files into the same plot for comparison

    Args:
        file_paths (list[str]): List of file paths to extract the metrics from
    """


print(extract_results("baseline", "results/baseline-results.json"))
