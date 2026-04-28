"""
ui/app.py — Streamlit Document Q&A Interface.

Features:
  - Drag-and-drop PDF / TXT / MD / CSV upload
  - Multi-document selection for scoped queries
  - Q&A with full source citation cards (filename, page, relevance %)
  - Confidence meter and answer type badge
  - Suggested follow-up questions
  - Q&A session history
  - Chunking strategy selector and top-k slider

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BACKEND = "http://localhost:8000"

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Q&A · RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .source-card {
        background: #1e2130; border-radius: 8px; padding: 14px 16px;
        border-left: 3px solid #00b4d8; margin-bottom: 10px;
    }
    .source-header { font-size: 0.82rem; color: #90e0ef; font-weight: 600; }
    .source-content { font-size: 0.88rem; color: #caf0f8; margin-top: 6px; }
    .confidence-high  { color: #06d6a0; font-weight: 700; }
    .confidence-mid   { color: #ffd166; font-weight: 700; }
    .confidence-low   { color: #ef8c8c; font-weight: 700; }
    .badge-rag      { background: #023e8a; color: #90e0ef; padding: 2px 8px;
                       border-radius: 4px; font-size: 0.75rem; }
    .badge-fallback { background: #6c757d; color: #fff; padding: 2px 8px;
                       border-radius: 4px; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    try:
        r = getattr(requests, method)(f"{BACKEND}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach backend. Start it with: `uvicorn app.main:app --reload`")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


def confidence_html(score: float) -> str:
    pct = f"{score:.0%}"
    cls = "confidence-high" if score >= 0.6 else ("confidence-mid" if score >= 0.35 else "confidence-low")
    label = "High" if score >= 0.6 else ("Medium" if score >= 0.35 else "Low")
    return f'<span class="{cls}">{label} ({pct})</span>'


# ── Session state ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {question, answer, sources, confidence, badge}
if "documents" not in st.session_state:
    st.session_state.documents = []


def refresh_documents():
    data = api("get", "/api/v1/documents")
    if data:
        st.session_state.documents = data.get("documents", [])


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Document Q&A")
    st.caption("Powered by RAG · OpenAI · FAISS")

    # Health check
    health = api("get", "/health")
    if health:
        st.success(f"✅ Backend online  v{health.get('version', '?')}")
        col1, col2 = st.columns(2)
        col1.metric("Documents", health.get("documents_indexed", 0))
        col2.metric("Chunks", health.get("total_chunks", 0))
    else:
        st.error("❌ Backend offline")

    st.divider()

    # ── Upload section ────────────────────────────────────────────────────────
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF, TXT, Markdown, or CSV",
        type=["pdf", "txt", "md", "csv"],
        label_visibility="collapsed",
    )
    strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "sentence", "fixed"],
        index=0,
        help="recursive: best for mixed content · sentence: best for prose · fixed: fastest",
    )

    if st.button("📥 Ingest Document", use_container_width=True, disabled=not uploaded_file):
        with st.spinner("Processing …"):
            result = api(
                "post", "/api/v1/documents/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")},
                data={"chunking_strategy": strategy},
            )
        if result:
            st.success(f"✅ {result['num_chunks']} chunks indexed from '{result['filename']}'")
            refresh_documents()

    st.divider()

    # ── Query settings ────────────────────────────────────────────────────────
    st.subheader("⚙️ Query Settings")
    top_k   = st.slider("Top-K chunks", 1, 15, 5)
    use_mmr = st.checkbox("MMR re-ranking (diversity)", value=True,
                           help="Maximal Marginal Relevance: balances relevance with diversity")

    # ── Document selector ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Document Scope")
    refresh_documents()

    if st.session_state.documents:
        doc_options = {d["filename"]: d["doc_id"] for d in st.session_state.documents}
        selected_filenames = st.multiselect(
            "Query these documents (empty = all)",
            options=list(doc_options.keys()),
            default=[],
        )
        selected_doc_ids = [doc_options[f] for f in selected_filenames] or None

        # Delete button
        if selected_filenames and len(selected_filenames) == 1:
            if st.button("🗑 Delete selected", use_container_width=True):
                did = doc_options[selected_filenames[0]]
                result = api("delete", f"/api/v1/documents/{did}")
                if result:
                    st.success(f"Deleted '{selected_filenames[0]}'")
                    refresh_documents()
                    st.rerun()
    else:
        selected_doc_ids = None
        st.info("No documents uploaded yet.")

    st.divider()
    if st.button("🗑 Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📚 Document Q&A with RAG")
st.caption(
    "Ask questions about your uploaded documents. "
    "Answers are grounded in your documents with source citations."
)

# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input("Ask a question about your documents …")

if question:
    with st.spinner("Searching and generating answer …"):
        payload = {
            "question": question,
            "top_k": top_k,
            "use_mmr": use_mmr,
        }
        if selected_doc_ids:
            payload["doc_ids"] = selected_doc_ids

        result = api("post", "/api/v1/qa/ask", json=payload)

    if result:
        st.session_state.chat_history.append(result)

        # Follow-up suggestions (best-effort, non-blocking)
        try:
            fu_resp = requests.post(
                f"{BACKEND}/api/v1/qa/ask",
                json={
                    "question": f"Suggest 3 follow-up questions for: {question}",
                    "top_k": 3,
                },
                timeout=10,
            )
        except Exception:
            pass


# ── Chat history ──────────────────────────────────────────────────────────────

if not st.session_state.chat_history:
    st.info("👈 Upload a document and ask a question to get started.")
else:
    for entry in reversed(st.session_state.chat_history):

        # Question bubble
        with st.chat_message("user"):
            st.write(entry["question"])

        # Answer bubble
        with st.chat_message("assistant"):
            badge_cls = "badge-rag" if entry.get("answer_type") == "rag" else "badge-fallback"
            badge_txt = "RAG Answer" if entry.get("answer_type") == "rag" else "Fallback"
            conf_score = entry.get("confidence", 0.0)

            st.markdown(
                f'<span class="{badge_cls}">{badge_txt}</span>&nbsp;&nbsp;'
                f'Confidence: {confidence_html(conf_score)}&nbsp;&nbsp;'
                f'<span style="color:#6c757d;font-size:0.8rem;">'
                f'⏱ {entry.get("latency_ms", 0):.0f} ms | '
                f'Model: {entry.get("model_used", "?")}</span>',
                unsafe_allow_html=True,
            )
            st.write(entry["answer"])

            # Source citation cards
            sources = entry.get("sources", [])
            if sources:
                st.markdown("**📖 Sources used:**")
                for src in sources:
                    page_str   = f" · Page {src['page_number']}" if src.get("page_number") else ""
                    section_str = f" · {src['section']}" if src.get("section") else ""
                    rel_str    = f"{src['relevance_score']:.0%}"
                    preview    = src["content"][:350] + ("…" if len(src["content"]) > 350 else "")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">'
                        f'📄 {src["filename"]}{page_str}{section_str} &nbsp;·&nbsp; Relevance: {rel_str}'
                        f'</div>'
                        f'<div class="source-content">{preview}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Metadata row
            st.caption(
                f"Question ID: `{entry.get('question_id', 'N/A')}` · "
                f"Sources: {len(sources)} · "
                f"Timestamp: {entry.get('timestamp', '')[:19]}"
            )


# ── Documents table (collapsible) ─────────────────────────────────────────────

if st.session_state.documents:
    with st.expander(f"📋 Indexed Documents ({len(st.session_state.documents)})", expanded=False):
        import pandas as pd
        rows = [
            {
                "Filename":   d["filename"],
                "Type":       d["file_type"].upper(),
                "Chunks":     d["num_chunks"],
                "Size (KB)":  round(d["file_size_bytes"] / 1024, 1),
                "Uploaded":   str(d["uploaded_at"])[:16],
                "Doc ID":     d["doc_id"],
            }
            for d in st.session_state.documents
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
