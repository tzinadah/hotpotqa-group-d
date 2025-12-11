"""
This experiment is to answer the full dataset using the expert template
as it's the most effecient with respect to results
"""

from hotpotqa_group_d.config import Model, expert_template
from hotpotqa_group_d.pipelines import RAG_answer

if __name__ == "__main__":

    # Expert expirement
    RAG_answer(
        "predictions/full-answers.json",
        template=expert_template,
        model=Model.MEDIUM,
        top_k=30,
    )
