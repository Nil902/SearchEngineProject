# RAG-Based AI Search System — Starter Project

This is a runnable starting point for your **final project**. It is intentionally
minimal: every piece works today with zero API keys, and every piece has a clearly
marked upgrade path so you can grow it into your real final submission.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and ask a question
like *"How does content-based filtering rank items?"* against the sample documents.

## What's already working

- **Ingestion** (`rag/ingest.py`) — loads `.txt` files and splits them into
  overlapping chunks.
- **Retrieval** (`rag/embed_store.py`) — vectorizes chunks with TF-IDF (the same
  technique from the Week 14 lab) and ranks them by cosine similarity.
- **Generation** (`rag/generate.py`) — an `extractive` mode that needs no API key,
  plus an `llm` mode stub ready for you to wire up a real model.
- **Interface** (`app.py`) — a Streamlit search UI: query box, answer panel, and an
  expandable, scored list of source chunks. Sidebar controls `top_k` and answer mode.

## Project structure

```
final_project_starter/
├── app.py                  # Streamlit interface
├── requirements.txt
├── data/sample_docs/        # replace with your own domain's documents
└── rag/
    ├── ingest.py            # load + chunk documents
    ├── embed_store.py       # vectorize + similarity search
    └── generate.py          # turn retrieved chunks into an answer
```

## Your upgrade path (this is most of the final project)

1. **Swap in your own dataset.** Replace the files in `data/sample_docs/` with your
   chosen domain (product docs, papers, articles, notes — anything text-heavy).
2. **Upgrade retrieval from TF-IDF to real embeddings.** In `rag/embed_store.py`,
   replace `TfidfVectorizer` with a sentence-embedding model (e.g.
   `sentence-transformers`) or an embeddings API. Keep the `VectorStore.build` /
   `.query` interface the same so `app.py` doesn't need to change.
3. **Move to a real vector database once your corpus is large**, e.g. FAISS or
   Chroma, for faster search than the in-memory cosine similarity used here.
4. **Wire up an LLM in `rag/generate.py`'s `llm_answer`** so answers are generated
   and grounded in the retrieved context, with citations back to source documents.
5. **Extend the interface**: file upload for new documents, highlighting matched
   terms, a settings panel for chunk size, response latency display, conversation
   history, etc.
6. **Add an evaluation section**: a small set of test queries with expected sources,
   and a short write-up of what worked, what didn't, and why.

## Why start from this

The retrieval mechanics here (chunk \u2192 vectorize \u2192 cosine similarity \u2192 rank) are
exactly the ones you practiced in the Week 14 content-based filtering lab, just
applied to document chunks instead of movies. Getting this skeleton running today
means every future class session is an upgrade to a working system, not a fresh
start.
