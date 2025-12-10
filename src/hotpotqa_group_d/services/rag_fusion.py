import time
import uuid

import chromadb
from chromadb.utils.embedding_functions import MistralEmbeddingFunction
import os
from hotpotqa_group_d.config import Env, Model
from hotpotqa_group_d.services import parse_data

from mistralai import Mistral
from hotpotqa_group_d.services.rag import chunk_doc, embed_data

def rrf_fuse(ranked_lists, rrf_k = 60, top_k = 5):
    """
    reciprocal rank fusion implementation for retrieved documents

    args:
        ranked_lists (list): List of ranked lists of (ids, documents, metadatas)
        rrf_k (int): parameter for normalising the scores
        top_k (int): number of top documents to return

    returns:
        fused_list (list): fused list of (id, document, metadata, score)    
    """

    scores = {}
    payload = {}

    for items in ranked_lists:
        for rank, (doc_id, doc, meta) in enumerate(items, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in payload:
                payload[doc_id] = (doc, meta)

    # sort by score reversed to get highest scores first
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_list = []
    for doc_id, score in fused:
        doc, meta = payload[doc_id]
        fused_list.append((doc_id, doc, meta, score))

    return fused_list


def generate_subqueries(question, n = 6, model = "mistral-large-latest"):
    """
    Generate subqueries for retrieval using Mistral
    
    args:
        question (str): the query
        n (int): number of subqueries to generate
        model (str): mistral model to use

    returns:
        output (list): list of subqueries
    """

    api_key = os.getenv("MISTRAL_KEY")
    if not api_key:
        return [question]

    client = Mistral(api_key=api_key)

    prompt = (
        f"Generate {n} diverse and relevant search subqueries to retrieve passages for answering the question.\n"
        f"Question: {question}\n"
        "Rules:\n"
        "- One query per line\n"
        "- No numbering, no bullets\n"
        "- Keep each query short (<= 12 words)\n"
    )

    for j in range(5):
        try:
            resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception:
            print(f"API Error {j+1} retrying {5-j-1} times")
        else:
            # if all retries failed, skip this variant
            continue

    text = resp.choices[0].message.content or ""
    queries = [q.strip() for q in text.splitlines() if q.strip()]

    output = []
    seen = set()
    for q in [question] + queries:
        if q not in seen:
            seen.add(q)
            output.append(q)
    return output[:max(1, n)]


def retrieve_docs_fusion(
    query,
    top_k = 5,
    embeddings_path = "./chroma_db",
    n_variants = 6,
    per_variant_k = 8,
    rrf_k = 50,
):
    """
    Retrieval using Misral to generate subqueries and RRF to fuse and rank results

    args:
        query (str): the main query
        k (int): number of top documents to return
        embeddings_path (str): path to chroma database
        n_variants (int): number of subqueries to generate
        per_variant_k (int): number of top documents to retrieve per subquery
        rrf_k (int): RRF normalisation parameter
    
    returns:
        list of top k documents as a string
    """

    chroma_client = chromadb.PersistentClient(path=embeddings_path)

    collection = chroma_client.get_collection(
        name="test_collection",
        embedding_function=MistralEmbeddingFunction(
            model=Model.EMBED.value, api_key_env_var="MISTRAL_KEY"
        ),
    )

    queries = generate_subqueries(query, n=n_variants)

    ranked_lists = []

    for q in queries:
        # Retry loop
        for j in range(5):
            try:
                results = collection.query(
                    query_texts=[q],
                    n_results=per_variant_k,
                    include=["documents", "metadatas"],  # ids are always included
                )
                break
            except Exception:
                print(f"API Error {j+1} retrying {5-j-1} times")
        else:
            # if all retries failed, skip this variant
            continue

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        ranked_lists.append([(ids[i], docs[i], metas[i]) for i in range(len(ids))])

    fused = rrf_fuse(ranked_lists, rrf_k=rrf_k, top_k=top_k)

    # Build context block
    context_pieces = []
    for i, (_id, doc, meta, score) in enumerate(fused, start=1):
        title = meta.get("title", "UNKNOWN")
        chunk_num = meta.get("chunk", "UNKNOWN")
        context_pieces.append(f"{i}. [Title: {title}] , [Chunk: {chunk_num}], [RRF: {score:.4f}] {doc}")

    return "\n".join(context_pieces)


