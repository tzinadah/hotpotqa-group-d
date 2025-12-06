import uuid

import chromadb
from chromadb.utils.embedding_functions import MistralEmbeddingFunction

from hotpotqa_group_d.config import Env, Model
from hotpotqa_group_d.services import parse_data


def embed_data(
    data_path="data/hotpot_dev_fullwiki_v1.json", db_path="./chroma_db", batch_size=10
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
            model=Model.EMBED, api_key_env_var="MISTRAL_KEY"
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
            collection.add(documents=docs, metadatas=metas, ids=ids)
        print(f"Added batch {i} - {i+len(batch)} ({len(docs)} docs)")
