"""Retrieval layer: rank document chunks by relevance to a query.

This is stage (ii) of the RAG pipeline. It implements the BM25 ranking
function (Robertson & Zaragoza, 2009) from scratch — the standard bag-of-words
baseline used by search engines. Implementing it directly (rather than
importing a library) makes the mathematics easy to present and defend in the
project report, and keeps the system dependency-free.

The class exposes the same interface (`index`, `search`) that an
embedding-based retriever (e.g. sentence-transformers + ChromaDB) would, so
the vector-based upgrade discussed in the report is a drop-in replacement.
"""

import math
import re
from collections import Counter

# Standard BM25 hyperparameters from the literature.
K1 = 1.5  # term-frequency saturation
B = 0.75  # document-length normalisation

_WORD = re.compile(r"[a-z0-9]+")

# Common English words that carry no topical signal.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with which what when where who how not can also""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self._doc_tokens: list[Counter] = []
        self._doc_lengths: list[int] = []
        self._avg_length = 0.0
        self._idf: dict[str, float] = {}

    def index(self, chunks: list[str]) -> None:
        """Build the inverted statistics needed to score queries."""
        self.chunks = chunks
        self._doc_tokens = [Counter(tokenize(c)) for c in chunks]
        self._doc_lengths = [sum(c.values()) for c in self._doc_tokens]
        n = len(chunks)
        self._avg_length = (sum(self._doc_lengths) / n) if n else 0.0

        # Inverse document frequency: rare terms are more informative.
        doc_freq: Counter = Counter()
        for tokens in self._doc_tokens:
            doc_freq.update(tokens.keys())
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 6) -> list[tuple[int, float]]:
        """Return (chunk_index, score) for the top_k most relevant chunks."""
        query_terms = tokenize(query)
        scores = []
        for i, tokens in enumerate(self._doc_tokens):
            score = 0.0
            length_norm = 1 - B + B * (self._doc_lengths[i] / self._avg_length or 1.0)
            for term in query_terms:
                tf = tokens.get(term, 0)
                if tf == 0:
                    continue
                score += self._idf.get(term, 0.0) * (tf * (K1 + 1)) / (tf + K1 * length_norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]

    def spread_sample(self, top_k: int = 6) -> list[tuple[int, float]]:
        """Sample chunks evenly across the document (unique indices).

        Used when no topic is given, or as a fallback when BM25 finds nothing.
        Indices are spaced from first to last chunk so long documents are not
        collapsed onto the opening pages.
        """
        n = len(self.chunks)
        if n == 0 or top_k <= 0:
            return []
        if n <= top_k:
            return [(i, 0.0) for i in range(n)]
        # Inclusive endpoints; avoid round() collisions that duplicate indices.
        indices: list[int] = []
        for i in range(top_k):
            idx = int(round(i * (n - 1) / (top_k - 1)))
            indices.append(idx)
        # Repair any residual duplicates by shifting forward.
        seen: set[int] = set()
        unique: list[int] = []
        for idx in indices:
            while idx in seen and idx + 1 < n:
                idx += 1
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        # If still short, fill gaps from unused indices
        if len(unique) < top_k:
            for i in range(n):
                if i not in seen:
                    unique.append(i)
                    seen.add(i)
                if len(unique) >= top_k:
                    break
        unique.sort()
        return [(i, 0.0) for i in unique[:top_k]]
