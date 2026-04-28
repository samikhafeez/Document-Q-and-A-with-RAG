"""
prompts.py — All prompt templates for the RAG pipeline.

Design principles:
  1. Ground the LLM strictly in retrieved context — prevents hallucination.
  2. Handle definitional questions differently from analytical ones.
     Definitional ("What is X?") → 1–3 sentence direct answer.
     Analytical (how, why, compare) → structured paragraphs.
  3. Clean inline citation format: (filename, p.N) — compact, not repeated.
  4. Explicit weak-context instruction: if sources don't support the answer, say so.
  5. Never expose relevance scores to the LLM — it leads to score-anchored wording.
  6. Hard cap on context: only pass the chunks that are genuinely relevant.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from app.models import SourceChunk


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG QA prompt
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer the user's question using ONLY the context passages provided below.

── ANSWER STYLE ──────────────────────────────────────────────────────────────
• Definitional questions ("What is X?", "Define X", "What does X mean?", "Who is X?"):
  Answer in 1–3 sentences. Lead directly with the definition or explanation.
  Do NOT add background, context, or related information unless the source text
  explicitly provides it.

• Analytical questions (how something works, comparisons, causes, implications):
  Use concise, focused paragraphs. Cover only what the context directly supports.
  Do not speculate or extend beyond the source material.

── CITATION FORMAT ───────────────────────────────────────────────────────────
Cite inline immediately after the information you use, like this:
  (EchoLeak.pdf, p.3)      ← when a page number is available
  (EchoLeak.pdf)           ← when no page number is available
Do NOT list all sources at the end. Do NOT repeat the same citation.
Do NOT fabricate filenames or page numbers.

── STRICT RULES ──────────────────────────────────────────────────────────────
1. Use ONLY the context passages below. Never add outside knowledge or general facts.
2. If the context does not contain enough information to answer the question, say:
   "The uploaded documents do not contain enough information to answer this question."
3. If the context is only partially relevant, answer only what IS directly supported,
   then add on a new line: "Note: the sources only partially address this question."
4. Do NOT repeat the same point using different wording from multiple chunks.
5. For definition questions, stop after the direct answer — do not pad with loosely
   related content just because it appears in the same document.

CONTEXT:
{context}
"""

RAG_HUMAN_PROMPT = "Question: {question}"

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human",  RAG_HUMAN_PROMPT),
])


# ─────────────────────────────────────────────────────────────────────────────
# Weak-context prompt — used when retrieval is present but confidence is low
# ─────────────────────────────────────────────────────────────────────────────
# This fires when confidence is between _WEAK_CONFIDENCE_THRESHOLD and
# _LOW_CONFIDENCE_THRESHOLD (i.e. we found SOMETHING but it's not strong).
# It explicitly tells the LLM to be conservative about what it claims.

WEAK_CONTEXT_SYSTEM_PROMPT = """\
You are a careful document assistant. Some context passages were found, but they may not directly answer the question.

RULES:
1. Answer ONLY from the context below — no outside knowledge.
2. If the context touches on the question only indirectly, say what the sources DO say
   and note that they do not fully address the question.
3. Do not expand, infer, or add information not present in the passages.
4. Use the citation format: (filename, p.N) inline.
5. If the context is genuinely not useful, say:
   "The uploaded documents do not contain enough information to answer this question."

CONTEXT:
{context}
"""

WEAK_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", WEAK_CONTEXT_SYSTEM_PROMPT),
    ("human",  "Question: {question}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Fallback prompt (no relevant context found)
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful document assistant. "
        "No relevant passages were found in the uploaded documents for this question. "
        "Tell the user clearly that the documents don't contain this information, "
        "and suggest they upload relevant documents or rephrase the question. "
        "Do not attempt to answer from general knowledge."
    )),
    ("human", "{question}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context_string(
    sources: list[SourceChunk],
    max_chunks: int = 4,
    min_score: float = 0.28,
) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Key decisions:
    - Only include chunks above min_score (removes noise that slipped through retrieval)
    - Cap at max_chunks (4) — stuffing all 5 weak chunks hurts more than it helps
    - Do NOT expose the relevance score to the LLM (it causes score-anchored wording
      and pushes the model to mention weak sources it otherwise would have ignored)
    - Use the same citation format as the prompt instructs: "filename, p.N"
      so the LLM can copy-paste citations exactly

    Args:
        sources   : Pre-filtered, sorted SourceChunk list from the retriever.
        max_chunks: Hard cap on context passages sent to the LLM (default 4).
        min_score : Per-chunk minimum relevance score (default 0.28).
                    Chunks below this are silently excluded from the context.
    """
    if not sources:
        return "No relevant context found."

    # Apply per-chunk minimum score filter and hard cap
    relevant = [s for s in sources if s.relevance_score >= min_score][:max_chunks]

    if not relevant:
        return "No sufficiently relevant context found."

    blocks = []
    for i, src in enumerate(relevant, 1):
        # Citation header matches exactly the format instructed in the prompt
        page_info = f", p.{src.page_number}" if src.page_number is not None else ""
        header = f"[{i}] {src.filename}{page_info}"
        blocks.append(f"{header}\n{src.content}")

    return "\n\n---\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Suggested follow-up questions prompt
# ─────────────────────────────────────────────────────────────────────────────

FOLLOWUP_PROMPT = PromptTemplate.from_template("""\
Given this question and answer about a document, suggest 3 short follow-up \
questions the user might want to ask next. Return only the questions, \
one per line, no numbering.

Question: {question}
Answer: {answer}
""")
