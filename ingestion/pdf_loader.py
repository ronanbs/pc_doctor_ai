"""
pdf_loader.py
Extracts text content from PDF documents (e.g. motherboard manuals) using
PyMuPDF (fitz). Handles multi-page docs and preserves page numbers for
citation purposes later in the pipeline.
"""

import os
import fitz  # PyMuPDF


def load_pdf(filepath: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.

    Returns a list of dicts: [{"page": int, "text": str}, ...]
    Page numbers are 1-indexed to match how humans reference PDF pages.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PDF not found: {filepath}")

    doc = fitz.open(filepath)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})

    doc.close()
    print(f"[pdf_loader] Extracted {len(pages)} non-empty pages from {filepath}")
    return pages


def load_pdf_as_single_text(filepath: str) -> str:
    """
    Convenience wrapper: returns the whole PDF as one string, with
    page-break markers preserved so downstream chunking can still
    recover approximate page numbers if needed.
    """
    pages = load_pdf(filepath)
    combined = "\n\n".join(f"[PAGE {p['page']}]\n{p['text']}" for p in pages)
    return combined


def save_extracted_text(filepath: str, save_dir: str = "data/raw") -> str:
    """
    Extract a PDF's text and save it as a .txt file for the chunking
    pipeline to pick up, matching the format used by scrapers.py.
    """
    text = load_pdf_as_single_text(filepath)
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(save_dir, f"{base_name}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[pdf_loader] Saved extracted text -> {out_path}")
    return out_path


if __name__ == "__main__":
    # Example usage — point this at your motherboard manual PDF
    example_pdf = "data/raw/motherboard/MAG_X870E_CARBON_WIFI_manual.pdf"
    if os.path.exists(example_pdf):
        save_extracted_text(example_pdf, save_dir="data/raw/motherboard")
    else:
        print(f"[pdf_loader] No file found at {example_pdf} — place your PDF there first.")
