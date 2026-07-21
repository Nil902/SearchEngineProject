"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Upgrade path (for your final project):
- Add PDF/HTML/Markdown loaders (e.g. pypdf, BeautifulSoup) alongside plain .txt
- Swap the naive word-count chunker below for a sentence- or token-aware chunker
- Store document metadata (source URL, author, date) alongside each chunk
"""

import os
import re
from dataclasses import dataclass
from typing import List

TEXT_EXTENSIONS = (".txt", ".md")


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str


def _parse_front_matter(raw: str, fallback_title:str) -> tuple:
    title, body = fallback_title, raw  # default if there is no header
    if raw.startswith("---"):    # check if the file have header
        parts = raw.split("---", 2)   # split ['', header, body]
        if len(parts) == 3:
            front, body = parts[1], parts[2]    # seperate the header and body
            match = re.search(r'^title:\s*"?(.+?)"?\s*$', front, re.MULTILINE)
            if match:
                title = match.group(1).strip() # use the header title
    return title, body.strip()


def _clean_markdown(text: str) -> str:
    """Strip Markdown syntax so embeddings and displayed passages read as prose.

    Removes the noise (link URLs, code-fence markers, heading/emphasis symbols)
    that otherwise pollutes retrieval — e.g. long go.sum dumps surfacing for
    unrelated questions.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)   # fenced code blocks
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          # images -> drop
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # links -> keep text
    text = re.sub(r"`([^`]*)`", r"\1", text)                   # inline code ticks
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"[*_>]", "", text)                          # emphasis / quotes
    text = re.sub(r"[ \t]+", " ", text)                        # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                     # collapse blank lines
    return text.strip()

def load_documents(folder: str) -> List[dict]:
    """Load every .txt/.md file in `folder` into {"title": ..., "text": ...} dicts."""
    docs = []
    for filename in sorted(os.listdir(folder)):  # sorted = stable order
        if not filename.endswith(TEXT_EXTENSIONS):    # skip images, etc
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        # Fallback title if a file has no front-matter: turn the filename into
        # words, e.g. "go-slices.md" -> "Go Slices".
        fallback_title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        title, text = _parse_front_matter(raw, fallback_title)
        docs.append({"title": title, "text": _clean_markdown(text)})
    return docs


# Split on sentence-ending punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_RE.split(text.replace("\n", " "))
    return [s.strip() for s in parts if s.strip()]


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Sentence-aware chunking.

    Packs whole sentences into chunks of up to ~`chunk_size` words, carrying the
    last `overlap` words of each chunk into the next so ideas that straddle a
    boundary stay retrievable from either side. Keeping sentence boundaries
    intact gives cleaner, more coherent passages than a blind word-count window.
    """
    if overlap >= chunk_size:   # guard so the carried-over context can't fill a whole chunk
        overlap = chunk_size - 1
    sentences = _split_sentences(text)
    if not sentences:   # empty document -> no chunk
        return []
    chunks: List[str] = []
    current: List[str] = []   # words accumulated for the chunk in progress
    for sentence in sentences:
        words = sentence.split()
        # A single sentence longer than the budget can't be packed — hard-split it.
        if len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))
            continue
        # Adding this sentence would overflow the budget -> close the current chunk,
        # then seed the next one with `overlap` words of trailing context.
        if current and len(current) + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunk_records(docs: List[dict], chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            records.append(Chunk(chunk_id=f"{doc['title']}::{i}", doc_title=doc["title"], text=piece))
    return records
