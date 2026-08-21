"""
app.py
PC Doctor AI — Streamlit frontend.

Users pick a troubleshooting category from a selector, then chat with a
RAG-grounded assistant scoped to that category's knowledge base.
"""

import streamlit as st

from categories import CATEGORIES, get_category_labels, label_to_key
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

# --- Chat state ---
# Keep a separate message history per category so switching categories
# doesn't mix unrelated conversations together.
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {key: [] for key in CATEGORIES}

history = st.session_state.chat_histories[selected_key]

# --- Render existing messages ---
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# --- Chat input ---
user_input = st.chat_input(f"Ask a {category_info['label']} question...")

if user_input:
    # Show user message immediately
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                result = answer_question(selected_key, category_info["label"], user_input)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for src in result["sources"]:
                            st.write(f"- {src}")
                history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                error_msg = (
                    "Sorry, something went wrong answering that question. "
                    f"Details: {e}"
                )
                st.error(error_msg)
                history.append({"role": "assistant", "content": error_msg, "sources": []})

# --- Sidebar: knowledge base status ---
with st.sidebar:
    st.header("About")
    st.write(
        "PC Doctor AI answers questions using a curated knowledge base "
        "per category, retrieved via a Chroma vector store and grounded "
        "with source citations."
    )
    st.write("**Categories:**")
    for key, info in CATEGORIES.items():
        st.write(f"- {info['label']}")

    st.divider()
    if st.button("Clear this category's chat"):
        st.session_state.chat_histories[selected_key] = []
        st.rerun()
