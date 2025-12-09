from hotpotqa_group_d.evaluation import extract_results
from hotpotqa_group_d.evaluation.visualisation import plot_metrics

if __name__ == "__main__":

    # Extract results
    baseline_results = extract_results("baseline", "results/baseline-results.json")
    medium_rag_results = extract_results(
        "medium rag", "results/medium-model-results.json"
    )

    # Plot
    plot_metrics([baseline_results, medium_rag_results], "plots/medium-rag.pdf")
