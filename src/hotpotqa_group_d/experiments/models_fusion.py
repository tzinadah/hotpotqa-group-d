"""
This expiremnt is to try different models than the baseline which
used the small-latest mistral model
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import RAG_answer, RAG_answer_fusion
from hotpotqa_group_d.services.rag_fusion import retrieve_docs_fusion
from hotpotqa_group_d.config import (
    Model,
    blackmail_template,
    expert_template,
    polite_template,
)

if __name__ == "__main__":
    # Medium model
    RAG_answer_fusion(
        "results/medium-model-fusion.json",
        embeddings_path="./chroma_db",
        template=expert_template,
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
        rrf_k=100
    )

