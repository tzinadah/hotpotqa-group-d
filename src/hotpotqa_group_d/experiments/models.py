"""
This expiremnt is to try different models than the baseline which
used the small-latest mistral model
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import RAG_answer

if __name__ == "__main__":
    # Small model
    RAG_answer(
        "predictions/small-model.json",
        embeddings_path="./chroma_db",
        model=Model.SMALL,
        sample_size=100,
        top_k=30,
    )

    # Medium model
    RAG_answer(
        "predictions/medium-model.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
        top_k=30,
    )

    # Large model
    RAG_answer(
        "predictions/large-model.json",
        embeddings_path="./chroma_db",
        model=Model.LARGE,
        sample_size=100,
        top_k=30,
    )
