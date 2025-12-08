import uuid

import chromadb
from chromadb.utils.embedding_functions import MistralEmbeddingFunction

from hotpotqa_group_d.config import Env, Model
from hotpotqa_group_d.services import parse_data


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
            for j in range(5):
                try:
                    collection.add(documents=docs, metadatas=metas, ids=ids)
                    break
                except Exception:
                    print(f"API Error {j+1} retrying {5-j-1} times")

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

    # Retry loop
    for j in range(5):
        try:
            # Retrieval based on cosine similarity
            results = collection.query(query_texts=[query], n_results=k)
            break
        except Exception:
            print(f"API Error {j+1} retrying {5-j-1} times")

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context block
    context_pieces = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        title = meta.get("title", "UNKNOWN")
        context_pieces.append(f"{i}. [Title: {title}] {doc}")

    context = "\n".join(context_pieces)

    return context
