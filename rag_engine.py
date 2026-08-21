"""
rag_engine.py
Core RAG generation logic: retrieves relevant chunks for a query,
injects them into an anchor prompt, calls the OpenAI chat model, and
returns an answer with source citations.
"""

import config
from ingestion.vector_store import query_category, get_openai_client


ANCHOR_PROMPT_TEMPLATE = """You are PC Doctor AI, a focused technical assistant that helps users \
troubleshoot PC hardware and software problems in the category: {category_label}.

Answer ONLY using the CONTEXT provided below, which comes from official documentation \
and trusted troubleshooting references. Do not use outside knowledge or guess.

Rules:
- If the answer is fully or partially supported by the context, answer clearly and cite \
  which source(s) you used (by filename, shown in brackets after each context chunk).
- If the context does not contain enough information to answer, say so plainly and do NOT \
  fabricate an answer. Suggest what category or search might help instead.
- If the question is unrelated to PC hardware/software troubleshooting (out-of-domain), \
  politely decline and explain that you only handle {category_label} troubleshooting.
- Keep answers practical and step-by-step where relevant.

CONTEXT:
{context_block}

USER QUESTION:
{question}

ANSWER (with citations in the form [source: filename]):"""


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks into a labeled context block for the prompt."""
    if not hits:
        return "(No relevant context was found in the knowledge base.)"

    blocks = []
    for hit in hits:
        blocks.append(f"[source: {hit['source']}]\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_question(category_key: str, category_label: str, question: str) -> dict:
    """
    Full RAG pipeline for one turn: retrieve -> build prompt -> generate.

    Returns: {"answer": str, "sources": list[str], "hits": list[dict]}
    """
    hits = query_category(category_key, question, top_k=config.TOP_K)
    context_block = build_context_block(hits)

    prompt = ANCHOR_PROMPT_TEMPLATE.format(
        category_label=category_label,
        context_block=context_block,
        question=question,
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer_text = response.choices[0].message.content

    unique_sources = sorted(set(h["source"] for h in hits))

    return {
        "answer": answer_text,
        "sources": unique_sources,
        "hits": hits,
    }
