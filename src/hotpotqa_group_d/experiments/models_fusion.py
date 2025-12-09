"""
This expiremnt is to try different models than the baseline which
used the small-latest mistral model
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import RAG_answer
from hotpotqa_group_d.services.rag_fusion import retrieve_docs_fusion

if __name__ == "__main__":
    # Medium model
    RAG_answer(
        "results/medium-model-fusion.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
        rag_method=retrieve_docs_fusion,
    )

