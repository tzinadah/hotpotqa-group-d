"""
This expiremnt is for self-reflection
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import two_step_self_reflection_answer, three_step_self_reflection_answer
from hotpotqa_group_d.config import (
    Model,
    blackmail_template,
    expert_template,
    polite_template,
)

if __name__ == "__main__":
    # Two step self -reflection with medium model
    two_step_self_reflection_answer(
        "results/medium-model-reflection.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

    # Two step self -reflection with large model
    two_step_self_reflection_answer(
        "results/large-model-reflection.json",
        embeddings_path="./chroma_db",
        model=Model.LARGE,
        sample_size=100,
        top_k=30,
    )
    
    # Three step self -reflection with medium model
    three_step_self_reflection_answer(
        "results/medium-model-reflection2.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

 