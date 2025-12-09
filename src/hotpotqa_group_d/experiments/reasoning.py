"""
Experiment to test both in prompt reasoning and reasoning models
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import RAG_answer

if __name__ == "__main__":
    # In prompt reasoning
    # RAG_answer(
    #     "results/in-prompt-reasoning.json",
    #     model=Model.MEDIUM,
    #     sample_size=100,
    #     template=reasoning_template,
    #     top_k=30,
    # )

    # Reasoning model
    RAG_answer(
        "results/reasoning-model.json", model=Model.REASONING, sample_size=100, top_k=1
    )
