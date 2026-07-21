"""
Vector store: turn chunks into vectors and support similarity search over them.

This uses sentence-transformers ("all-MiniLM-L6-v2") to turn each chunk into a
384-dimensional embedding, then ranks chunks against a query with cosine
similarity. The whole index lives in memory.

Two optimizations keep the live demo fast and the results clean:
- Embeddings are cached to disk (keyed by model + chunk contents), so a repeat
  run loads the matrix instead of re-encoding every chunk. The heavy embedding
  model is loaded lazily, so a cache hit skips loading it until the first query.
- Retrieval can optionally use MMR (maximal marginal relevance) to trade a little
  raw similarity for diversity, avoiding three near-duplicate chunks in the top-k.

Upgrade path (for your final project):
- Swap the in-memory cosine_similarity search below for FAISS or Chroma once your
  chunk count grows past a few thousand.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = ".cache"


def _fingerprint(model_name: str, texts: List[str]) -> str:
    """A short, stable hash of the model + every chunk text.

    If the documents, chunking, or model change, the fingerprint changes and the
    stale cache is bypassed automatically.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")            # separator so chunk boundaries matter
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


class VectorStore:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None   # loaded lazily
        self.matrix: Optional[np.ndarray] = None            # one row per chunk
        self.chunks: List[Chunk] = []                       # kept so we can return them

    @property
    def model(self) -> SentenceTransformer:
        """Load the embedding model on first use (a cache hit may skip it entirely)."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, chunks: List[Chunk]) -> None:
        """Embed every chunk once, up front, reusing a disk cache when possible."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        if not texts:                          # nothing to index
            self.matrix = None
            return

        cache_path = os.path.join(CACHE_DIR, f"emb_{_fingerprint(self.model_name, texts)}.npy")
        if os.path.exists(cache_path):         # cache hit -> load, don't re-encode
            self.matrix = np.load(cache_path)
            return

        self.matrix = self.model.encode(       # pass every chunk through the transformer
            texts,
            batch_size=32,                     # embed 32 at a time (speed)
            show_progress_bar=True,
            normalize_embeddings=True,         # length-1 vectors -> cosine == dot product
        )
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_path, self.matrix)       # save for next run

    def query(self, query_text: str, top_k: int = 3,
              use_mmr: bool = False, fetch_k: int = 20,
              lambda_mult: float = 0.5) -> List[Tuple[Chunk, float]]:
        """Embed the query and return the top_k most similar (chunk, score) pairs.

        With use_mmr=True, first pull `fetch_k` candidates by similarity, then greedily
        select a diverse subset (lambda_mult balances relevance vs. novelty).
        """
        if self.matrix is None or len(self.chunks) == 0:   # nothing indexed
            return []
        # Put the query into the same 384-d space as the chunks.
        query_vec = self.model.encode([query_text], normalize_embeddings=True)
        # Compare the query to every chunk at once -> one score per chunk.
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        if not use_mmr:
            # argsort sorts ascending; [::-1] flips to descending, [:top_k] takes best.
            ranked_idx = np.argsort(scores)[::-1][:top_k]
            return [(self.chunks[i], float(scores[i])) for i in ranked_idx]

        # MMR: start from the fetch_k most similar candidates, then pick greedily.
        candidates = list(np.argsort(scores)[::-1][:fetch_k])
        selected: List[int] = []
        while candidates and len(selected) < top_k:
            if not selected:
                best = candidates[0]           # most relevant first
            else:
                sel_matrix = self.matrix[selected]
                best, best_val = candidates[0], -np.inf
                for c in candidates:
                    # Redundancy = how similar this candidate is to what we already picked.
                    redundancy = float(np.max(cosine_similarity(self.matrix[c:c + 1], sel_matrix)))
                    val = lambda_mult * float(scores[c]) - (1 - lambda_mult) * redundancy
                    if val > best_val:
                        best_val, best = val, c
            selected.append(best)
            candidates.remove(best)
        return [(self.chunks[i], float(scores[i])) for i in selected]
