"""
vector_store.py
Handles embedding text chunks via the OpenAI API and storing/querying them
in per-category persistent Chroma collections.
"""

import os
import sys
import time
from io import BytesIO

# allow running as a script from ingestion/ as well as importing from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from openai import OpenAI
from pypdf import PdfReader

import config
from categories import CATEGORIES
from ingestion.clean_and_chunk import clean_text, chunk_text, process_category_folder


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


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from a Streamlit uploaded PDF or TXT file."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return file_bytes.decode("utf-8")


def get_collection_count(category_key: str) -> int:
    """Return how many chunks are stored for a category."""
    collection = get_or_create_collection(category_key)
    return collection.count()


def index_uploaded_document(
    category_key: str,
    text: str,
    source_name: str,
    batch_size: int = 50,
) -> int:
    """
    Chunk, embed, and upsert an uploaded document into the category's
    Chroma collection. Returns the number of chunks indexed.
    """
    cleaned = clean_text(text)
    chunk_strings = chunk_text(
        cleaned,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    if not chunk_strings:
        return 0

    chunks = [
        {"text": chunk, "source": source_name, "chunk_index": i}
        for i, chunk in enumerate(chunk_strings)
    ]

    collection = get_or_create_collection(category_key)
    upload_id = int(time.time() * 1000)
    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [f"{category_key}_upload_{upload_id}_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "category": category_key,
            }
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

    return total_indexed


def index_url(category_key: str, url: str) -> int:
    """Fetch a URL, extract text, and index it into ChromaDB."""
    from ingestion.scrapers import ScrapeError, scrape_url_text

    cleaned_url = url.strip()
    try:
        text = scrape_url_text(cleaned_url)
    except ScrapeError as exc:
        raise ValueError(str(exc)) from exc

    if not text:
        raise ValueError(f"Could not fetch or extract text from: {cleaned_url}")

    return index_uploaded_document(category_key, text, cleaned_url)


def build_all_collections():
    """Build/refresh Chroma collections for every category."""
    for category_key in CATEGORIES:
        print(f"\n=== Building collection: {category_key} ===")
        build_collection_for_category(category_key)


def semantic_search(
    category_key: str,
    query: str,
    top_k: int | None = None,
    max_distance: float | None = None,
) -> list[dict]:
    """
    Semantic search: embed the query and retrieve the most similar chunks
    from the category's Chroma collection.

    Returns hits sorted by relevance (lowest distance first):
        [{"text": str, "source": str, "distance": float, "chunk_index": int}, ...]
    """
    top_k = top_k or config.TOP_K
    max_distance = max_distance if max_distance is not None else config.MAX_RETRIEVAL_DISTANCE

    collection = get_or_create_collection(category_key)
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where={"category": category_key},
    )

    hits = []
    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        if max_distance is not None and dist > max_distance:
            continue
        hits.append({
            "text": doc,
            "source": meta.get("source", "unknown") if meta else "unknown",
            "distance": dist,
            "chunk_index": meta.get("chunk_index", -1) if meta else -1,
        })

    return hits


def query_category(category_key: str, query: str, top_k: int = None) -> list[dict]:
    """Backward-compatible alias for semantic_search."""
    return semantic_search(category_key, query, top_k=top_k)


if __name__ == "__main__":
    build_all_collections()
