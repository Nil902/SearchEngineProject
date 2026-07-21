"""
RAG-Based AI Search System — starter interface.

Run with:
    streamlit run app.py

This gives you a working, end-to-end demo today: document loading, TF-IDF based
retrieval, and an extractive answer — all wired into a real web interface. Build
your final project by upgrading each piece (see the TODOs in rag/embed_store.py
and rag/generate.py) without needing to touch this file's overall structure.
"""

import time

import streamlit as st

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/sample_docs"

st.set_page_config(page_title="RAG Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


store, docs, chunks = load_store()

with st.sidebar:
    st.header("Settings")
    # top_k = how many chunks to retrieve. This is the required tunable control.
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    # Let the user switch between the extractive fallback and real LLM answers.
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode needs GROQ_API_KEY set.")
    # MMR trades a little raw similarity for diversity, avoiding near-duplicate chunks.
    use_mmr = st.checkbox("Diverse results (MMR)", value=False,
                          help="Reduce near-duplicate chunks among the retrieved sources.")
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents \u2192 **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

# ---- Main panel: header, query box, answer, sources --------------------------
st.title("🔎 RAG-Based AI Search System")
st.caption("Ask a question about the indexed documents below.")

query = st.text_input("Your question", placeholder="e.g. How does content-based filtering rank items?")
search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    start = time.perf_counter()
    retrieved = store.query(query, top_k=top_k, use_mmr=use_mmr)
    answer = generate_answer(query, retrieved, mode=mode)
    elapsed = time.perf_counter() - start

    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Answered in {elapsed:.2f}s \u00b7 {mode} mode \u00b7 {len(retrieved)} sources")

    st.subheader("Sources")
    for chunk, score in retrieved:
        with st.expander(f"{chunk.doc_title}  \u00b7  similarity {score:.0%}"):
            st.write(chunk.text)
elif search_clicked:
    st.warning("Type a question first.")
