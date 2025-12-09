from hotpotqa_group_d.evaluation import extract_results
from hotpotqa_group_d.evaluation.visualisation import plot_metrics

if __name__ == "__main__":

    # Extract results
    baseline_results = extract_results("baseline", "results/baseline-results.json")

    small_rag_results = extract_results("small rag", "results/small-model-results.json")
    medium_rag_results = extract_results(
        "medium rag", "results/medium-model-results.json"
    )
    large_rag_results = extract_results("large rag", "results/large-model-results.json")

    blackmail_template_results = extract_results(
        "blackmail template", "results/blackmail-prompt-results.json"
    )
    polite_template_results = extract_results(
        "polite template", "results/polite-prompt-results.json"
    )
    expert_template_results = extract_results(
        "expert template", "results/expert-prompt-results.json"
    )

    prompt_reasoning_results = extract_results(
        "in-prompt reasoning", "results/in-prompt-reasoning-results.json"
    )
    model_reasoning_results = extract_results(
        "model reasoning", "results/reasoning-model-results.json"
    )

    # Plot RAG comparison
    plot_metrics([baseline_results, medium_rag_results], "plots/rag-comparison.pdf")
