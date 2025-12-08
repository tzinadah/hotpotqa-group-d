import time
import uuid

import chromadb
from chromadb.utils.embedding_functions import MistralEmbeddingFunction
import os
from typing import Dict, Any, List, Tuple
from hotpotqa_group_d.config import Env, Model
from hotpotqa_group_d.services import parse_data

from mistralai import Mistral


def chunk_doc(text, chunk_size=100, overlap=20):
    """
    Service to chunk a document to parts with respect to their length
    and with overlap

    Args:
        text (str): Document or text to chunk
        chunk_size (int): Number of maximum words per chunk
        overlap (int): Number of characters in between chunks

    Returns:
        chunks (list[str]): List of chunks
    """

    words = text.split()
    # Step to iterate over
    step = chunk_size - overlap

    chunks = list()

    for i in range(0, len(words), step):

        # Get a slice of words that represents the chunk
        chunk_list = words[i : i + chunk_size]

        # Convert back to string
        chunk = " ".join(chunk_list)

        chunks.append(chunk)

    return chunks


def embed_data(
    data_path="./data/hotpot_dev_fullwiki_v1.json", db_path="./chroma_db", batch_size=10
):
    """
    Service to embed the context paragraphs from dev full wiki

    Args:
        data_path (str): Path for wiki data
        db_path (str): Path for embeddings to be stored
        batch_size (int): Number of documents processed at once
    """

    client = chromadb.PersistentClient(path=db_path)
    env = Env()

    # get collection or create collection if it doesn't exist
    collection = client.get_or_create_collection(
        name="test_collection",
        embedding_function=MistralEmbeddingFunction(
            model=Model.EMBED.value, api_key_env_var="MISTRAL_KEY"
        ),
    )

    data = parse_data(data_path)

    # Set to keep track of duplicates
    processed_docs = set()

    # Process data in batches
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        docs = []
        metas = []
        ids = []
        # Extract context from each item in the batch
        for item in batch:
            context = item.get("context", [])

            # Ensure context is a non-empty list
            if isinstance(context, list) and context:

                # Each entry in context is expected to be a list where the first element is the title
                # and the second element is a paragraph
                for entry in context:
                    if not (isinstance(entry, list) and len(entry) >= 2):
                        continue

                    title = entry[0]
                    paragraph = entry[1]
                    text = " ".join(sentence.strip() for sentence in paragraph)

                    if title not in processed_docs:
                        processed_docs.add(title)

                        # Chunk text
                        chunks = chunk_doc(text, chunk_size=100, overlap=20)

                        for idx, chunk in enumerate(chunks):
                            docs.append(chunk)
                            metas.append(
                                {"title": title, "chunk": f"{ idx + 1 }/{len(chunks)}"}
                            )
                            ids.append(str(uuid.uuid4()))

        if docs:
            # Retry loop for faulty api responses
            for j in range(10):
                try:
                    collection.add(documents=docs, metadatas=metas, ids=ids)
                    break
                except Exception:
                    print(f"API Error {j+1} retrying {5-j-1} times")
                    # Wait a sec for API faults
                    time.sleep(0.5)

        print(f"Added batch {i} - {i+len(batch)} ({len(docs)} docs)")


def rrf_fuse(
    ranked_lists: List[List[Tuple[str, str, Dict[str, Any]]]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> List[Tuple[str, str, Dict[str, Any], float]]:
    """
    ranked_lists: list of ranked results per query variant:
      [[(id, doc, meta), (id, doc, meta), ...],  ...]
    returns: [(id, doc, meta, fused_score), ...] sorted best->worst
    """
    scores: Dict[str, float] = {}
    payload: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for hits in ranked_lists:
        for rank, (doc_id, doc, meta) in enumerate(hits, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in payload:
                payload[doc_id] = (doc, meta or {})

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_id, payload[doc_id][0], payload[doc_id][1], score) for doc_id, score in fused]


def generate_query_variants_mistral(question: str, n: int = 6, model: str = "mistral-small-latest") -> List[str]:
    """
    Uses Mistral chat completion to create diverse search queries.
    Falls back to just [question] if no API key is set.
    """
    # Your embedding code uses api_key_env_var="MISTRAL_KEY"
    # Mistral SDK examples default to MISTRAL_API_KEY, so we accept either.
    api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("MISTRAL_KEY")
    if not api_key:
        return [question]

    client = Mistral(api_key=api_key)

    prompt = (
        f"Generate {n} diverse search queries to retrieve passages for answering the question.\n"
        f"Question: {question}\n"
        "Rules:\n"
        "- One query per line\n"
        "- No numbering, no bullets\n"
        "- Keep each query short (<= 12 words)\n"
    )

    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    queries = [q.strip() for q in text.splitlines() if q.strip()]

    # ensure original is included + dedupe
    out = []
    seen = set()
    for q in [question] + queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[: max(1, n)]


def retrieve_docs(
    query: str,
    k: int = 5,
    embeddings_path: str = "./chroma_db",
    n_variants: int = 6,
    per_variant_k: int = 8,
    rrf_k: int = 60,
):
    """
    RAG-Fusion retrieval:
    - generate query variants
    - query chroma per variant
    - fuse results with RRF
    - return context string
    """
    chroma_client = chromadb.PersistentClient(path=embeddings_path)

    collection = chroma_client.get_collection(
        name="test_collection",
        embedding_function=MistralEmbeddingFunction(
            model=Model.EMBED.value, api_key_env_var="MISTRAL_KEY"
        ),
    )

    queries = generate_query_variants_mistral(query, n=n_variants)

    ranked_lists: List[List[Tuple[str, str, Dict[str, Any]]]] = []

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

    fused = rrf_fuse(ranked_lists, rrf_k=rrf_k, top_k=k)

    # Build context block
    context_pieces = []
    for i, (_id, doc, meta, score) in enumerate(fused, start=1):
        title = meta.get("title", "UNKNOWN")
        chunk_num = meta.get("chunk", "UNKNOWN")
        context_pieces.append(f"{i}. [Title: {title}] , [Chunk: {chunk_num}], [RRF: {score:.4f}] {doc}")

    return "\n".join(context_pieces)
