# PC Doctor AI

A RAG-based chat assistant for PC hardware and software troubleshooting.
Built as the final project for AI and ML Level 2.

## Scope

PC Doctor AI helps users diagnose and troubleshoot PC hardware and software
issues — including BSODs and Windows stop-code errors, GPU/driver failures,
motherboard/BIOS/power issues, and Linux kernel or driver-level crashes.
The assistant retrieves grounded answers from a curated knowledge base per
category, with source citations for every diagnostic claim.

## Categories

- **Windows / BSOD & Stop Codes** — Microsoft Bug Check Code Reference, boot
  troubleshooting docs
- **GPU & Drivers** — NVIDIA driver troubleshooting, DDU clean-reinstall guide
- **Motherboard / BIOS / Power** — MSI MAG X870E Carbon WiFi manual, XMP/RAM
  stability guide
- **Linux / Kernel Issues** — Arch Wiki NVIDIA page, Arch Wiki kernel panic page

Each category has its own persistent Chroma DB collection, so retrieval stays
scoped to the selected topic.

## Architecture

```
User picks category (Streamlit selectbox)
        |
        v
Query embedded (OpenAI text-embedding-3-small)
        |
        v
Top-K chunks retrieved from that category's Chroma collection
        |
        v
Chunks injected into an anchor prompt (rag_engine.py)
        |
        v
OpenAI chat model generates a grounded, cited answer
```

## Project structure

```
pc_doctor_ai/
├── app.py                   # Streamlit chat UI + category selector
├── rag_engine.py             # Anchor prompt + generation logic
├── categories.py              # Category definitions & Chroma collection mapping
├── config.py                   # API keys, model names, chunk params
├── ingestion/
│   ├── scrapers.py             # Web scraping (Microsoft Docs, Arch Wiki, etc.)
│   ├── pdf_loader.py            # PDF text extraction (motherboard manual)
│   ├── clean_and_chunk.py        # Text cleaning + character chunking
│   └── vector_store.py            # Embedding + Chroma collection build/query
├── data/
│   ├── raw/                        # Scraped/extracted source text (gitignored)
│   └── chroma_db/                    # Persistent vector store (gitignored)
├── requirements.txt
└── .env.example
```

## Setup

1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your OpenAI API key
4. Add source PDFs to `data/raw/<category>/` and/or fill in URLs in
   `ingestion/scrapers.py`'s `SOURCE_URLS`
5. Build the knowledge base:
   ```
   python -m ingestion.scrapers        # scrape web sources
   python -m ingestion.pdf_loader      # extract PDF sources (edit path first)
   python -m ingestion.vector_store    # chunk, embed, and index everything
   ```
6. Run the app: `streamlit run app.py`

## Stress-testing protocol (Phase 4)

Each category is audited with four question types:
- **Direct** — fact-checking against the specific source text
- **Vague/Implicit** — testing conceptual connections
- **Adversarial** — attempts to break persona or bypass the category scope
- **Out-of-Domain** — confirming the assistant declines irrelevant topics
