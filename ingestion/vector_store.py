"""
vector_store.py
Handles embedding text chunks via the OpenAI API and storing/querying them
in per-category persistent Chroma collections.
"""

import os
import sys

# allow running as a script from ingestion/ as well as importing from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from openai import OpenAI

import config
from categories import CATEGORIES
from ingestion.clean_and_chunk import process_category_folder


_client = None
_openai_client = None


def get_chroma_client():
    global _client
    if _client is None:
        os.makedirs(config.CHROMA_DB_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    return _client


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using the OpenAI embeddings API."""
    client = get_openai_client()
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def get_or_create_collection(category_key: str):
    """Get (or create) the Chroma collection for a given category."""
    if category_key not in CATEGORIES:
        raise ValueError(f"Unknown category: {category_key}")

    client = get_chroma_client()
    collection_name = CATEGORIES[category_key]["collection_name"]
    return client.get_or_create_collection(name=collection_name)


def build_collection_for_category(category_key: str, batch_size: int = 50) -> int:
    """
    Process a category's raw text files into chunks, embed them, and
    upsert them into that category's Chroma collection.

    Returns the number of chunks indexed.
    """
    cat = CATEGORIES[category_key]
    chunks = process_category_folder(
        cat["raw_folder"],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    if not chunks:
        print(f"[vector_store] No chunks to index for {category_key}")
        return 0

    collection = get_or_create_collection(category_key)

    total_indexed = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [f"{category_key}_{i + j}" for j in range(len(batch))]
        metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"], "category": category_key}
            for c in batch
        ]

        embeddings = embed_texts(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_indexed += len(batch)
        print(f"[vector_store] Indexed {total_indexed}/{len(chunks)} chunks for {category_key}")

    return total_indexed


def build_all_collections():
    """Build/refresh Chroma collections for every category."""
    for category_key in CATEGORIES:
        print(f"\n=== Building collection: {category_key} ===")
        build_collection_for_category(category_key)


def query_category(category_key: str, query: str, top_k: int = None) -> list[dict]:
    """
    Embed a query and retrieve the top_k most relevant chunks from a
    category's Chroma collection.

    Returns a list of dicts: [{"text": str, "source": str, "distance": float}, ...]
    """
    top_k = top_k or config.TOP_K
    collection = get_or_create_collection(category_key)

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "distance": dist,
        })

    return hits


if __name__ == "__main__":
    build_all_collections()
