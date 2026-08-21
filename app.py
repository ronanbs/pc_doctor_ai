"""
app.py
PC Doctor AI — Streamlit frontend.

Users pick a troubleshooting category from a selector, then chat with a
RAG-grounded assistant scoped to that category's knowledge base.
"""

import streamlit as st

from categories import CATEGORIES, get_category_labels, label_to_key
import config
from ingestion.vector_store import (
    extract_text_from_upload,
    get_collection_count,
    index_uploaded_document,
    index_url,
)
from rag_engine import answer_question

st.set_page_config(page_title="PC Doctor AI", page_icon="🖥️", layout="centered")

st.title("🖥️ PC Doctor AI")
st.caption("RAG-grounded PC hardware & software troubleshooting assistant")

# --- Category selector ---
labels = get_category_labels()
selected_label = st.selectbox("Choose a troubleshooting category:", labels)
selected_key = label_to_key(selected_label)
if selected_key is None:
    st.error("Invalid category selected.")
    st.stop()

category_info = CATEGORIES[selected_key]

st.write(f"*{category_info['description']}*")
st.divider()


def render_source_attributions(result: dict) -> None:
    """Show citation map and optionally the retrieved chunks."""
    source_map = result.get("source_map") or {}
    hits = result.get("hits") or []

    if source_map:
        with st.expander("📎 Source attributions"):
            for num in sorted(source_map):
                st.markdown(f"**[Source {num}]** → `{source_map[num]}`")

    if hits and st.session_state.get("show_retrieved_sources", True):
        with st.expander("🔎 Retrieved context"):
            for hit in hits:
                source_num = hit.get("source_number", "?")
                distance = hit.get("distance")
                dist_label = f" (distance: {distance:.4f})" if distance is not None else ""
                st.markdown(f"### [Source {source_num}]{dist_label}")
                st.caption(f"From: `{hit.get('source', 'unknown')}`")
                st.write(hit.get("text", ""))
                st.divider()


# --- Chat state ---
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {key: [] for key in CATEGORIES}

history = st.session_state.chat_histories[selected_key]

# --- Render existing messages ---
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("rag_enabled"):
            render_source_attributions(msg)

# --- Chat input ---
user_input = st.chat_input(f"Ask a {category_info['label']} question...")

if user_input:
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        rag_enabled = st.session_state.get("rag_enabled", True)
        top_k = st.session_state.get("top_k", config.TOP_K)

        spinner_text = (
            "🔎 Retrieving relevant sources..."
            if rag_enabled
            else "💬 Generating answer (RAG off)..."
        )

        with st.spinner(spinner_text):
            try:
                result = answer_question(
                    selected_key,
                    category_info["label"],
                    user_input,
                    use_rag=rag_enabled,
                    top_k=top_k,
                )
                st.markdown(result["answer"])

                if result.get("rag_enabled"):
                    st.caption(f"✅ Retrieved {len(result.get('hits', []))} relevant chunks.")
                else:
                    st.caption("🟡 RAG is off — no document sources were retrieved.")

                render_source_attributions(result)

                history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "source_map": result.get("source_map", {}),
                    "hits": result.get("hits", []),
                    "rag_enabled": result.get("rag_enabled", False),
                })
            except Exception as e:
                error_msg = (
                    "Sorry, something went wrong answering that question. "
                    f"Details: {e}"
                )
                st.error(error_msg)
                history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "source_map": {},
                    "hits": [],
                    "rag_enabled": False,
                })

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.write(
        "PC Doctor AI answers questions using a curated knowledge base "
        "per category, retrieved via semantic search (Chroma + OpenAI "
        "embeddings) and grounded with numbered source citations."
    )
    st.write("**Categories:**")
    for key, info in CATEGORIES.items():
        st.write(f"- {info['label']}")

    st.divider()
    st.header("RAG Settings")

    st.session_state.rag_enabled = st.checkbox(
        "🔎 Enable RAG (semantic search)",
        value=st.session_state.get("rag_enabled", True),
        help="When enabled, the app retrieves relevant chunks before generating an answer.",
    )

    st.session_state.top_k = st.slider(
        "📚 Chunks to retrieve",
        min_value=1,
        max_value=8,
        value=st.session_state.get("top_k", config.TOP_K),
        help="How many semantically similar chunks to pull from Chroma.",
    )

    st.session_state.show_retrieved_sources = st.checkbox(
        "👀 Show retrieved context",
        value=st.session_state.get("show_retrieved_sources", True),
    )

    if st.session_state.rag_enabled:
        st.success("🟢 RAG: Enabled")
    else:
        st.warning("🟡 RAG: Disabled")

    st.divider()

    st.header("Add to Knowledge Base")
    st.caption(
        f"Index documents into **{category_info['label']}** for RAG retrieval."
    )
    st.write(f"Chunks in this category: **{get_collection_count(selected_key)}**")

    page_url = st.text_input(
        "Documentation URL",
        placeholder="https://learn.microsoft.com/...",
        key=f"url_{selected_key}",
    )

    if st.button("Fetch URL & add to knowledge base"):
        if not page_url.strip():
            st.warning("Enter a URL first.")
        else:
            with st.spinner(f"Fetching and indexing {page_url}..."):
                try:
                    chunk_count = index_url(selected_key, page_url)
                    st.success(
                        f"Indexed **{chunk_count}** chunks from the URL into ChromaDB."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"URL indexing failed: {e}")

    uploaded_file = st.file_uploader(
        "Or upload a PDF / TXT file",
        type=["pdf", "txt"],
        key=f"upload_{selected_key}",
    )

    if st.button("Upload file to knowledge base"):
        if uploaded_file is None:
            st.warning("Choose a file first.")
        else:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    text = extract_text_from_upload(uploaded_file)
                    if not text.strip():
                        st.error("No text could be extracted from that file.")
                    else:
                        chunk_count = index_uploaded_document(
                            selected_key,
                            text,
                            uploaded_file.name,
                        )
                        st.success(
                            f"Indexed **{chunk_count}** chunks from "
                            f"`{uploaded_file.name}` into ChromaDB."
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.divider()
    if st.button("Clear this category's chat"):
        st.session_state.chat_histories[selected_key] = []
        st.rerun()
