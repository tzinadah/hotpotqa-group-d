"""
This expiremnt is for self-reflection
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import RAG_self_reflection_answer
from hotpotqa_group_d.config import (
    Model,
    blackmail_template,
    expert_template,
    polite_template,
)

if __name__ == "__main__":
    # SELF REFLECTION with clear template
    RAG_self_reflection_answer(
        "results/medium-model-reflection.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

 