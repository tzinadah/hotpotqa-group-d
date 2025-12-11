"""
This expiremnt is for self-fusion and the two-step reflection
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import two_step_self_reflection_answer, three_step_self_reflection_answer, self_reflection_fusion_RAG_answer
from hotpotqa_group_d.config import (
    Model,
    blackmail_template,
    expert_template,
    polite_template,
)

if __name__ == "__main__":
    # Two step self-reflection with medium model
   
    self_reflection_fusion_RAG_answer(
        "results/medium-model-fusion-two-step-reflection.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

    

 