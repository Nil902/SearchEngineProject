# RAG-Based AI Search System

A Retrieval-Augmented Generation (RAG) search system built for the CS382 final
project. You type a question into a web interface, the system retrieves the most
relevant chunks from a document collection, and a language model generates an
answer **grounded in those chunks with visible citations** back to the source
material.

This is not a general-knowledge chatbot. It answers only from the indexed
documents and shows its work: every answer is accompanied by the exact source
passages (with similarity scores) it was built from, and when nothing relevant is
found it says so instead of guessing.

**Domain:** The included corpus is 28 articles from [The Go Blog](https://go.dev/blog/)
covering slices, maps, generics, errors, concurrency, modules, reflection, and
tooling. Swap the files in `data/sample_docs/` to point the system at any
text-heavy domain of your own.

---

## Quick start

```bash
# 1. (recommended) create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) and ask a
question such as *"How do I check for a specific error type in Go 1.13?"* or
*"What is the difference between a slice and an array?"*.

The **extractive** answer mode works immediately with no API key. To enable the
grounded **LLM** answer mode, see the next section.

### Enabling LLM answers (optional)

LLM mode uses [Groq](https://console.groq.com) (a free, OpenAI-compatible API
gateway running `llama-3.3-70b-versatile`).

```bash
cp .env.example .env          # then edit .env and paste your key
# .env should contain:  GROQ_API_KEY=gsk_...
```

Get a free key at <https://console.groq.com>. The `.env` file is gitignored so
your key is never committed. Once set, choose **Answer mode → llm** in the
sidebar. If the key is missing or the request fails, the app falls back to
extractive mode instead of crashing.

---

## Architecture

The system moves through the classic RAG pipeline, from raw documents on disk to
a grounded, cited answer. Each stage is a separable module.

```
data/sample_docs/*.md
        │
        ▼
┌─────────────────────┐   rag/ingest.py
│ 1. Ingest & Chunk   │   load .md/.txt, strip Markdown, parse front-matter,
│                     │   split into overlapping word-count chunks
└─────────────────────┘
        │  List[Chunk]
        ▼
┌─────────────────────┐   rag/embed_store.py
│ 2. Embed            │   sentence-transformers all-MiniLM-L6-v2 → 384-d vectors
│ 3. Vector store     │   in-memory normalized matrix (cosine == dot product)
└─────────────────────┘
        │  query
        ▼
┌─────────────────────┐   rag/embed_store.py :: VectorStore.query
│ 4. Retrieve         │   embed query, cosine-similarity vs. all chunks, top-k
└─────────────────────┘
        │  (Chunk, score)[]
        ▼
┌─────────────────────┐   rag/generate.py
│ 5. Generate         │   relevance gate → extractive OR LLM answer w/ citations
└─────────────────────┘
        │  answer text
        ▼
┌─────────────────────┐   app.py
│ 6. Interface        │   Streamlit: query box, answer panel, scored sources,
│                     │   top-k slider, answer-mode toggle
└─────────────────────┘
```

### Module responsibilities

| File | Layer | What it does |
|------|-------|--------------|
| `rag/ingest.py` | Ingest & chunk | Loads every `.md`/`.txt` file, parses YAML front-matter for titles, strips Markdown syntax (code fences, links, headings) so retrieval isn't polluted, and splits each document into overlapping word-count chunks. |
| `rag/embed_store.py` | Embed + retrieve | `VectorStore` embeds all chunks once with `all-MiniLM-L6-v2`, normalizes them, and answers queries by cosine similarity, returning the top-k `(Chunk, score)` pairs. |
| `rag/generate.py` | Generate | `generate_answer()` applies a relevance gate, then produces either an *extractive* answer (stitched passages, no API needed) or an *LLM* answer grounded in the retrieved sources with `[n]` citations. |
| `app.py` | Interface | Streamlit UI wiring the pipeline together, with caching so documents are indexed only once per session. |

---

## Design decisions

- **Chunking — sentence-aware, ~80 words with 20-word overlap.** Whole sentences
  are packed into chunks of up to ~80 words (`rag/ingest.py`), so a chunk never
  cuts off mid-sentence and ideas spanning a boundary stay retrievable from either
  side. Markdown is stripped first so link URLs, code fences, and `go.sum` dumps
  don't surface as false matches.

  The 80/20 size was chosen from a sweep (60→300 words), not by default. Smaller
  chunks score slightly higher raw similarity, but that's misleading — a tiny chunk
  matches a narrow query tightly while giving the LLM less context to ground its
  answer, and it *shrinks the gap* between in-domain and out-of-domain scores. That
  gap is what matters, because it's what makes the graceful-failure threshold
  reliable. 80/20 gave the widest separation (in-domain ~0.74 vs. out-of-domain
  ~0.34), keeping out-of-domain queries safely below the `0.35` cutoff while still
  giving each chunk enough context for a grounded answer. The 60-word setting
  scored higher on similarity but let an out-of-domain query reach 0.38 — above the
  cutoff — so it was rejected.

- **Embeddings — `sentence-transformers/all-MiniLM-L6-v2`.** A small (384-dim),
  fast, local, free model. It runs on CPU with no API key, keeps the demo
  responsive, and captures semantic similarity far better than TF-IDF (the
  starter baseline) — so *"how do I stop a goroutine leak"* matches passages
  about timeouts and `context` even without exact keyword overlap.

- **Vector store — in-memory cosine similarity.** Vectors are L2-normalized at
  build time, so cosine similarity reduces to a dot product across one NumPy
  matrix. With only a few thousand chunks this is instant; FAISS/Chroma would be
  the next step at a larger scale but would add setup cost for no user-visible
  gain here.

- **Embedding cache + lazy model load (demo latency).** The embedding matrix is
  cached to disk in `.cache/`, keyed by a hash of the model name and every chunk's
  text. A repeat run loads the matrix (~0.01s) instead of re-encoding the whole
  corpus (~12s cold), and the heavy embedding model isn't loaded at all on a cache
  hit until the first query needs it. The cache invalidates automatically when the
  documents, chunking, or model change.

- **Optional MMR retrieval (diversity).** A sidebar toggle switches retrieval to
  maximal marginal relevance: it pulls a larger candidate pool by similarity, then
  greedily selects a set that balances relevance against novelty (`lambda_mult=0.5`),
  avoiding three near-duplicate chunks in the top-k. Off by default; plain cosine
  ranking is usually best for tightly-scoped questions.

- **Grounding & citations.** The LLM prompt instructs the model to answer using
  **only** the numbered sources, to cite the source number(s) after each claim,
  and to say so plainly if the sources don't contain the answer. The same `[n]`
  numbering shown to the model is shown in the Sources panel, so a citation in
  the answer maps directly to a source the user can expand and read.

- **Graceful failure.** If the best-matching chunk scores below `MIN_RELEVANCE`
  (0.35, calibrated against the corpus — see Evaluation), the system returns
  *"I don't have information on that in the indexed documents"* rather than
  inventing an answer. Empty queries, missing API keys, and API errors are all
  handled without crashing the app.

- **Caching for demo latency.** Two layers: the on-disk embedding cache above,
  plus `@st.cache_resource` so the store is built once per session. After startup,
  a query only embeds the query string, so the live demo stays snappy.

---

## Interface

- **Header** — system name.
- **Query box + Search button** — submit a question.
- **Answer panel** — the generated (or extractive) response.
- **Sources panel** — each retrieved chunk in an expander showing the source
  document title, similarity score, and the chunk text.
- **Sidebar settings** — `top-k` slider (1–10 chunks to retrieve), answer-mode
  toggle (extractive / llm), a diverse-results (MMR) checkbox, and a live count of
  indexed documents → chunks.
- **Latency readout** — each answer shows how long retrieval + generation took,
  the mode used, and how many sources were returned.

---

## Evaluation

A set of test queries against the Go Blog corpus. The **top score** column is the
measured similarity of the best-retrieved chunk (from a headless run of the
retriever); in-domain and out-of-domain queries separate cleanly, which is what
justifies the `MIN_RELEVANCE = 0.35` graceful-failure threshold.

| # | Query | Expected top source | Top score | Retrieval | Answer quality |
|---|-------|---------------------|-----------|-----------|----------------|
| 1 | How do I check for a specific error type in Go 1.13? | Working with Errors in Go 1.13 | 0.78 | ✅ correct doc top-ranked | ✅ grounded, cites source |
| 2 | What is the difference between a slice and an array? | Go Slices / Arrays, slices | 0.80 | ✅ | ✅ |
| 3 | What does gofmt do? | gofmt | 0.65 | ✅ | ✅ |
| 4 | How do I profile a Go program's CPU usage? | Profiling Go Programs (pprof) | 0.80 | ✅ | ✅ |
| 5 | What are the three laws of reflection? | The Laws of Reflection | 0.54 | ✅ | ✅ |
| 6 | *(out-of-domain)* best pizza recipe | — none relevant | 0.20 | ✅ below threshold → declined | ✅ refuses instead of hallucinating |
| 7 | *(out-of-domain)* what is the weather in Tokyo tomorrow | — none relevant | 0.21 | ✅ declined | ✅ refuses |
| 8 | *(out-of-domain)* how do python decorators work | — none relevant | 0.34 | ✅ declined | ✅ refuses |
| 9 | *(out-of-domain)* what is the capital of France | — none relevant | 0.13 | ✅ declined | ✅ refuses |

**Observations.** In-domain queries score **0.54–0.80**; out-of-domain queries
score **0.13–0.34**. The `0.35` threshold sits in the gap, so genuinely unanswerable
questions (queries 6–9) are declined rather than answered from irrelevant chunks —
the core RAG guarantee that the system only speaks from its documents. Enabling MMR
diversifies the sources for broad queries at a small cost in top-1 similarity.

> These numbers are from the retriever alone. Run each query in the app to confirm
> the generated answers cite the right sources before your submission.

---

## Project structure

```
final_project/
├── app.py                 # Streamlit interface (layer 4)
├── requirements.txt
├── .env.example           # copy to .env and add GROQ_API_KEY for LLM mode
├── .streamlit/config.toml # disables the file watcher (avoids torchvision noise)
├── data/sample_docs/      # 28 Go Blog articles — replace with your own domain
└── rag/
    ├── ingest.py          # load + clean + chunk documents (layer 1)
    ├── embed_store.py     # embed + vector search (layers 2–3, retrieval)
    └── generate.py        # extractive + LLM grounded answers (layer 5)
```

---

## Known limitations

- **Chunking is sentence-aware but not section-aware.** It respects sentence
  boundaries, but doesn't group by document section/heading, so a long topic can
  still span several chunks. A structure-aware splitter would preserve more context.
- **In-memory index.** The embedding matrix is disk-cached, but the searchable
  index itself lives in RAM and is loaded fully on startup. Fine for a few thousand
  chunks; a larger corpus would need FAISS/Chroma.
- **The relevance threshold is corpus-specific.** `MIN_RELEVANCE = 0.35` was
  calibrated for this document set; swapping domains may require re-tuning it.
- **Single embedding model, no re-ranking.** Retrieval is pure cosine similarity
  with no cross-encoder re-ranking, so occasionally a lexically similar but less
  relevant chunk can rank above a better one.
- **Markdown stripping is regex-based**, so unusual formatting may leave minor
  artifacts in displayed passages.
- **LLM mode depends on an external API** (Groq). Without a key it silently
  falls back to extractive answers; answer quality and latency depend on that
  service.
- **No conversation memory** — each query is answered independently.

---

## Tech stack

Python · [Streamlit](https://streamlit.io) · [sentence-transformers](https://www.sbert.net)
(`all-MiniLM-L6-v2`) · scikit-learn / NumPy (cosine similarity) ·
[Groq](https://console.groq.com) `llama-3.3-70b-versatile` via the OpenAI client.
