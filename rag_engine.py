"""
rag_engine.py
Phase 3 RAG pipeline: semantic search -> anchor prompt -> generation -> citations.

Steps:
  1. Semantic search (embed query, retrieve top-K chunks from Chroma)
  2. Build numbered context block for the anchor prompt
  3. Send system (anchor) + user messages to the chat model
  4. Return answer with a source map linking [Source N] to filenames/URLs
"""

from openai.types.chat import ChatCompletionMessageParam

import config
from ingestion.vector_store import get_openai_client, semantic_search


ANCHOR_SYSTEM_PROMPT = """You are PC Doctor AI, a focused technical assistant that helps users \
troubleshoot PC hardware and software problems in the category: {category_label}.

Your job is to answer the user's question using ONLY the retrieved context below, \
which comes from official documentation and trusted troubleshooting references.

IMPORTANT RULES:

1. Use the supplied context as your primary source of information.

2. Do not invent facts, steps, or error codes that are not supported by the context.

3. If the answer cannot be found in the supplied context, clearly say:
   "I don't have enough information in the knowledge base to answer that."
   Do NOT guess. Suggest what the user might search for or which category might help.

4. If the question is unrelated to PC hardware/software troubleshooting, politely decline \
   and explain that you only handle {category_label} troubleshooting.

5. Whenever you use information from a source, include a citation immediately after the claim.

6. Use ONLY this citation format:
   [Source 1]
   [Source 2]
   [Source 3]

7. Do not create fake source numbers. Only cite sources that appear in the context below.

8. Keep answers practical and step-by-step where relevant.

---------------------------------------------------
RETRIEVED CONTEXT
---------------------------------------------------

{context_block}

---------------------------------------------------
END OF CONTEXT
---------------------------------------------------"""


def build_context_block(hits: list[dict]) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks into a numbered context block for the anchor prompt.

    Each chunk is labeled [Source 1], [Source 2], etc. The hits list is updated
    in place with a source_number field for the citation map.
    """
    if not hits:
        return "(No relevant context was found in the knowledge base.)", hits

    blocks = []
    for i, hit in enumerate(hits):
        source_number = i + 1
        hit["source_number"] = source_number
        blocks.append(f"[Source {source_number}]\n{hit['text']}")

    return "\n\n".join(blocks), hits


def build_source_map(hits: list[dict]) -> dict[int, str]:
    """Map citation numbers to their original source filename or URL."""
    return {
        hit["source_number"]: hit["source"]
        for hit in hits
        if "source_number" in hit
    }


def build_anchor_prompt(category_label: str, context_block: str) -> str:
    """Build the augmented system prompt (anchor prompt) with injected context."""
    return ANCHOR_SYSTEM_PROMPT.format(
        category_label=category_label,
        context_block=context_block,
    )


def generate_answer(system_prompt: str, question: str) -> str:
    """Send the anchor prompt + user question to the chat model."""
    client = get_openai_client()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=config.CHAT_TEMPERATURE,
    )
    content = response.choices[0].message.content
    return content or ""


def answer_without_rag(category_label: str, question: str) -> dict:
    """Generation loop with RAG disabled — no retrieval, no citations."""
    client = get_openai_client()
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                f"You are PC Doctor AI, a PC troubleshooting assistant for "
                f"{category_label}. Answer helpfully, but note you are not "
                f"grounded in a document knowledge base for this response."
            ),
        },
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=config.CHAT_TEMPERATURE,
    )
    content = response.choices[0].message.content or ""
    return {
        "answer": content,
        "sources": [],
        "source_map": {},
        "hits": [],
        "rag_enabled": False,
    }


def answer_question(
    category_key: str,
    category_label: str,
    question: str,
    *,
    use_rag: bool = True,
    top_k: int | None = None,
) -> dict:
    """
    Full RAG generation loop for one turn.

    Returns:
        {
            "answer": str,
            "sources": list[str],       # unique source filenames/URLs cited
            "source_map": dict[int,str], # [Source N] -> filename/URL
            "hits": list[dict],          # retrieved chunks with metadata
            "rag_enabled": bool,
        }
    """
    if not use_rag:
        return answer_without_rag(category_label, question)

    # Step 1: Semantic search
    hits = semantic_search(category_key, question, top_k=top_k)

    # Step 2: Build numbered context block
    context_block, hits = build_context_block(hits)
    source_map = build_source_map(hits)

    # Step 3: Inject context into anchor prompt
    system_prompt = build_anchor_prompt(category_label, context_block)

    # Step 4: Generate grounded answer with citation instructions
    answer_text = generate_answer(system_prompt, question)

    unique_sources = sorted(set(h["source"] for h in hits))

    return {
        "answer": answer_text,
        "sources": unique_sources,
        "source_map": source_map,
        "hits": hits,
        "rag_enabled": True,
    }
