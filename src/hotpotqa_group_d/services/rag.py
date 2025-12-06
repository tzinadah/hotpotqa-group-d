import uuid

import chromadb
from chromadb.utils.embedding_functions import MistralEmbeddingFunction

from hotpotqa_group_d.config import Env, Model
from hotpotqa_group_d.services import parse_data


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
                    text = (
                        title
                        + ":"
                        + " ".join(sentence.strip() for sentence in paragraph)
                    )
                    docs.append(text)
                    metas.append({"title": title})
                    ids.append(str(uuid.uuid4()))

        if docs:
            # Retry loop for faulty api responses
            for j in range(5):
                try:
                    collection.add(documents=docs, metadatas=metas, ids=ids)
                    break
                except Exception:
                    print(f"API Error {i+1} retrying {5-i-1} times")

        print(f"Added batch {i} - {i+len(batch)} ({len(docs)} docs)")


def retrieve_docs(query, k=5, embeddings_path="./chroma_db"):
    """
    Method to retrieve top relevant docs

    Args:
        query (str): Text to compare against
        k (int): Number of top relevant docs to retrieve

    Return:
        context (str): Context string of top k relevant docs
    """

    chroma_client = chromadb.PersistentClient(path=embeddings_path)

    collection = chroma_client.get_collection(
        name="test_collection",
        embedding_function=MistralEmbeddingFunction(
            model=Model.EMBED.value, api_key_env_var="MISTRAL_KEY"
        ),
    )

    # Retrieval based on cosine similarity
    results = collection.query(query_texts=[query], n_results=k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context block
    context_pieces = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        title = meta.get("title", "UNKNOWN")
        context_pieces.append(f"{i}. [Title: {title}] {doc}")

    context = "\n".join(context_pieces)

    return context
