from hotpotqa_group_d.evaluation import extract_results
from hotpotqa_group_d.evaluation.visualisation import plot_metrics, spider_plot

if __name__ == "__main__":

    # Extract results
    baseline_results = extract_results("baseline", "results/baseline-results.json")

    small_rag_results = extract_results("small rag", "results/small-model-results.json")
    medium_rag_results = extract_results(
        "medium rag", "results/medium-model-results.json"
    )
    large_rag_results = extract_results("large rag", "results/large-model-results.json")

    blackmail_template_results = extract_results(
        "blackmail", "results/blackmail-prompt-results.json"
    )
    polite_template_results = extract_results(
        "polite", "results/polite-prompt-results.json"
    )
    expert_template_results = extract_results(
        "expert", "results/expert-prompt-results.json"
    )

    prompt_reasoning_results = extract_results(
        "in-prompt reasoning", "results/in-prompt-reasoning-results.json"
    )
    model_reasoning_results = extract_results(
        "model reasoning", "results/reasoning-model-results.json"
    )

    fusion_results = extract_results(
        "rag fusion", "results/medium-model-fusion-results.json"
    )

    two_step_reflection_results = extract_results(
        "2 step reflection", "results/medium-model-reflection-results.json"
    )
    three_step_reflection_results = extract_results(
        "3 step reflection", "results/medium-model-reflection2-results.json"
    )

    mixed_results = extract_results("Combined Methods", "results/medium-model-fusion-two-step-reflection-results.json")

    # Plot RAG comparison
    plot_metrics([baseline_results, medium_rag_results], "plots/rag-comparison.pdf")

    # Plot model differences
    plot_metrics(
        [small_rag_results, medium_rag_results, large_rag_results],
        "plots/model-comparison.pdf",
    )

    # Plot template differences
    plot_metrics(
        [
            medium_rag_results,
            blackmail_template_results,
            polite_template_results,
            expert_template_results,
        ],
        "plots/template-comparison.pdf",
    )

    # Plot resoaning differences
    plot_metrics(
        [medium_rag_results, prompt_reasoning_results, model_reasoning_results],
        "plots/reasoning-comparison.pdf",
    )

    # Plot rag fusion
    plot_metrics(
        [medium_rag_results, fusion_results],
        "plots/rag-fusion-comparison.pdf",
    )

    # Plot self reflection
    plot_metrics(
        [
            medium_rag_results,
            two_step_reflection_results,
            three_step_reflection_results,
        ],
        "plots/self-reflection-comparison.pdf",
    )

    # Plot mixed methods
    plot_metrics(
        [
            mixed_results,
            two_step_reflection_results,
            fusion_results,
        ],
        "plots/mixed-comparison.pdf",
    )

    # Plot all models
    spider_plot(
        [
            baseline_results,
            small_rag_results,
            medium_rag_results,
            large_rag_results,
            blackmail_template_results,
            polite_template_results,
            expert_template_results,
            model_reasoning_results,
            prompt_reasoning_results,
            fusion_results,
            two_step_reflection_results,
            three_step_reflection_results,
            mixed_results,
        ],
        "plots/all-models.pdf",
    )

    
