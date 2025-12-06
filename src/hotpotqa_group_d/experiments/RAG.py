"""
This expiremnt is to try the rag pipeline using the small-latest mistral model

"""

from hotpotqa_group_d.config import Model
from hotpotqa_group_d.pipelines import basic_answer, RAG_answer

if __name__ == "__main__":
    # RAG 
    RAG_answer("results/RAG_SMALL_MODEL.json", model=Model.SMALL, sample_size=100, top_k=10)

    