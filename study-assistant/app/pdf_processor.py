"""Document processing: extract text from lecture files and split it into chunks.

This is stage (i) of the RAG pipeline described in Chapter 3 of the project
report: PDF extraction and chunking.
"""

import io
import re

from pypdf import PdfReader

# Chunks are sized in words. Roughly 200 words per chunk keeps each chunk
# focused on one idea, and the overlap stops a sentence that straddles a
# boundary from being lost to both chunks.
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


def extract_text(filename: str, data: bytes) -> str:
    """Extract raw text from an uploaded .pdf or .txt file."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    elif filename.lower().endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported file type. Upload a .pdf, .txt or .md file.")

    # Collapse repeated whitespace that PDF extraction tends to produce.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-window chunks.

    A sliding window is the simplest chunking strategy that still preserves
    local context; more advanced strategies (per-heading, per-paragraph)
    are discussed as future work in the report.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    for start in range(0, len(words), step):
        window = words[start : start + CHUNK_SIZE_WORDS]
        if len(window) < 30 and chunks:
            # Tail too small to be a useful chunk on its own — merge into
            # the previous one instead of creating a fragment.
            chunks[-1] = chunks[-1] + " " + " ".join(window)
            break
        chunks.append(" ".join(window))
    return chunks
