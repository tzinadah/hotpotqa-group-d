"""
This expirement is to test different approaches with templating the prompt

1- Asking the LLM to act as an expert
2- Asking the LLM politely
3- Asking the LLM with emotional blackmail
"""

from hotpotqa_group_d.config import (
    Model,
    blackmail_template,
    expert_template,
    polite_template,
)
from hotpotqa_group_d.pipelines import RAG_answer

if __name__ == "__main__":

    # Expert expirement
    RAG_answer(
        "results/expert-prompt.json",
        template=expert_template,
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

    # Polite expirement
    RAG_answer(
        "results/polite-prompt.json",
        template=polite_template,
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

    # Blackmail expirement
    RAG_answer(
        "results/blackmail-prompt.json",
        template=blackmail_template,
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )
