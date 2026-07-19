"""
Generation: turn retrieved chunks + a query into a final answer.

Two modes are provided:
- "extractive" (default): no API key needed, works immediately. Just stitches
  together the retrieved chunks so you can verify retrieval quality before wiring
  up an LLM.
- "llm": calls an LLM to write a grounded answer from the retrieved context.
  This project uses Groq (OpenAI-compatible API). Set GROQ_API_KEY in a .env file
  to enable it — get a free key at https://console.groq.com.

  This is what separates a RAG system from a plain chatbot: the LLM is told to
  answer ONLY from the retrieved chunks and to CITE them -- so answers are grounded
  in your documents, not the model's own memory.
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv

from .ingest import Chunk

# Load key/value pairs from a local .env file into os.environ (if present).
load_dotenv()

MIN_RELEVANCE = 0.15 # If retrieved chunk scores below this, assume nothing is relevant


def _format_sources(retrieved: List[Tuple[Chunk, float]]) -> str:
    """
    Turn the retrieved chunks into a numbered block the LLM can read and cite:
        [1] Errors are values
        <chunk text...>

        [2] Working with Errors in Go 1.13
        <chunk text...>
    The numbers [1], [2] are what the model cites, AND what the UI shows next to
    each source -- so a citation in the answer maps to a source in the panel.
    """
    return "\n\n".join(
        f"[{i}] {chunk.doc_title}\n{chunk.text}"
        for i, (chunk, _) in enumerate(retrieved, 1)   # start numbering at 1
    )



def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """The real RAG answer: send query + chunks to the LLM, get a grounded reply."""
    if not retrieved:
        return "No relevant passages were found, so I can't answer from the documents."

    context = _format_sources(retrieved)          # the numbered sources block

    # The prompt is where grounding happens. Three instructions matter most:
    #   1. answer using ONLY these sources (no outside knowledge)
    #   2. cite the source number(s) after each claim
    #   3. if the sources don't answer it, SAY SO (don't hallucinate)
    prompt = (
        "You are a precise assistant. Answer the QUESTION using ONLY the numbered "
        "SOURCES below. After each claim, cite the source number(s) you used, e.g. "
        "[1] or [2][3]. If the sources do not contain the answer, say so plainly "
        "instead of guessing.\n\n"
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )

    from openai import OpenAI                 # imported here so the file loads

    # Groq is OpenAI-compatible: same client, just a different base_url + key.
    # The key is read from the .env file (GROQ_API_KEY=...), which is gitignored
    # so it never gets pushed to GitHub. Get a free key at https://console.groq.com
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return ("[LLM mode not configured] Set GROQ_API_KEY to enable grounded "
                "answers (free key at https://console.groq.com). Falling back to "
                "extractive mode:\n\n" + extractive_answer(query, retrieved))

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",  # Groq's OpenAI-compatible endpoint
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",          # free, capable Groq model
            max_tokens=600,                           # cap the answer length
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content            # the model's written answer
    except Exception as e:
        # Don't crash the app if the gateway is unreachable or the key is bad.
        return (f"[LLM request failed: {e}]\n\nFalling back to extractive mode:\n\n"
                + extractive_answer(query, retrieved))



def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]],
                    mode: str = "extractive") -> str:
    """
    The single entry point app.py calls. Two responsibilities:
      1. GRACEFUL FAILURE: bail out early if the best match is too weak.
      2. Dispatch to either the LLM answer or the extractive fallback.
    """
    # retrieved[0] is the top hit; [1] of that pair is its similarity score.
    if retrieved and retrieved[0][1] < MIN_RELEVANCE:
        return ("I don't have information on that in the indexed documents. "
                f"(Best match scored only {retrieved[0][1]:.2f}.)")
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)
