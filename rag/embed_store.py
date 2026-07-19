"""
Vector store: turn chunks into vectors and support similarity search over them.

This uses sentence-transformers ("all-MiniLM-L6-v2") to turn each chunk into a
384-dimensional embedding, then ranks chunks against a query with cosine
similarity. The whole index lives in memory.

Upgrade path (for your final project):
- Swap the in-memory cosine_similarity search below for FAISS or Chroma once your
  chunk count grows past a few thousand.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self, model_name: str = MODEL_NAME): # -> model_name = MODEL_NAME
        # Load embedding model once.
        self.model = SentenceTransformer(model_name)
        self.matrix = None   # hold the number of chunk
        self.chunks: List[Chunk] = []    # keep chunks so we can return them

    def build(self, chunks: List[Chunk]) -> None:
        """Embed every chunk once, up front, and cache the result matrix"""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.model.encode(  # This pass all the text into transformer to convert to vector
            texts,
            batch_size=32,              # embed 32 at a time ( speed )
            show_progress_bar=True,
            normalize_embeddings=True,  # scale each vector to length 1, which make cosine == dot product
            )

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Embed the query and return the top_k most similar ( chunk, score ) pairs"""
        if self.matrix is None:
            raise RuntimeError(" call build() before query().")
        if len(self.chunks) == 0:   # nothing was indexed -> nothing to return
            return []
        # Put the query into the same 384-d space as the chunks
        query_vec = self.model.encode([query_text], normalize_embeddings=True)
        # Compare the query to every chunk at once -> one score per chunk
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        # argsort sorts ascending; [::-1] flips to descending [:top_k] takes best.
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        # Return the actual Chunk obejects alongside their similarity scores.
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]
