from hotpotqa_group_d.config.consts import Model
from hotpotqa_group_d.pipelines.answering_pipelines import RAG_answer

if __name__ == "__main__":
    RAG_answer("/results/temp.json", "./chroma_db", Model.MEDIUM, sample_size=5)
