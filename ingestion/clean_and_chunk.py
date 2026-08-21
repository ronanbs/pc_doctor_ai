"""
clean_and_chunk.py
Normalizes raw text files (from scrapers.py or pdf_loader.py) and splits
them into overlapping character-based chunks suitable for embedding.

Character chunking (rather than token chunking) is used here per the
course's Phase 2 requirement ("implement character chunking").
"""

import os
import re
import glob


def clean_text(text: str) -> str:
    """Basic normalization: collapse whitespace, strip weird artifacts."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    chunk_size: max characters per chunk
    chunk_overlap: characters shared between consecutive chunks, to
                   preserve context across chunk boundaries
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break on a sentence/paragraph boundary near the end,
        # rather than mid-word, when possible.
        if end < text_len:
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "))
            if last_break > chunk_size * 0.5:  # only trim if it's not too short
                chunk = chunk[: last_break + 1]
                end = start + len(chunk)

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap  # step forward with overlap

    return chunks


def process_file(filepath: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[dict]:
    """
    Load a .txt file, clean it, chunk it, and return chunk metadata dicts:
    [{"text": str, "source": str, "chunk_index": int}, ...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    source_name = os.path.basename(filepath)
    return [
        {"text": chunk, "source": source_name, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]


def process_category_folder(
    folder: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Process every .txt file in a category's raw-data folder and return
    a combined list of chunk dicts, ready for embedding.
    """
    all_chunks = []
    txt_files = glob.glob(os.path.join(folder, "*.txt"))

    if not txt_files:
        print(f"[clean_and_chunk] No .txt files found in {folder}")
        return all_chunks

    for filepath in txt_files:
        chunks = process_file(filepath, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
        print(f"[clean_and_chunk] {os.path.basename(filepath)} -> {len(chunks)} chunks")

    print(f"[clean_and_chunk] Total chunks for {folder}: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Example: process the windows_bsod category folder
    example_folder = "data/raw/windows_bsod"
    if os.path.exists(example_folder):
        chunks = process_category_folder(example_folder)
        print(f"First chunk preview:\n{chunks[0]['text'][:300]}..." if chunks else "No chunks produced.")
    else:
        print(f"[clean_and_chunk] Folder not found: {example_folder}")
