"""
This expiremnt is to try different models than the baseline which
used the small-latest mistral model
"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import RAG_answer

if __name__ == "__main__":
    # Small model
    RAG_answer(
        "results/small-model.json",
        embeddings_path="./chroma_db",
        model=Model.SMALL,
        sample_size=100,
    )

    # Medium model
    RAG_answer(
        "results/medium-model.json",
        embeddings_path="./chroma_db",
        model=Model.MEDIUM,
        sample_size=100,
    )

    # Large model
    RAG_answer(
        "results/large-model.json",
        embeddings_path="./chroma_db",
        model=Model.LARGE,
        sample_size=100,
    )
