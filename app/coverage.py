"""Long-document coverage: scale retrieval and batch generation across sections.

Previously the pipeline always used a fixed top_k=6. Long lectures were mostly
ignored. This module:

1. Scales how many chunks to retrieve with document size and quiz length.
2. Diversifies topic search so hits are not all adjacent pages.
3. Splits selected chunks into ordered section batches so generation can
   cover the whole selection without stuffing everything into one prompt.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .retriever import BM25Retriever

# Hard caps keep prompt size and API cost bounded on very large uploads.
MIN_TOP_K = 6
MAX_TOP_K = 18
MAX_CHUNKS_PER_BATCH = 8  # chunks sent to Claude in one generation call


@dataclass
class RetrievalPlan:
    """What context the quiz generator will see (for logs / evaluation)."""

    strategy: str  # spread | bm25_diversified | all_chunks | baseline_none
    top_k: int
    chunk_indices: list[int]
    num_batches: int
    questions_per_batch: list[int]
    num_document_chunks: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_top_k(num_chunks: int, num_questions: int) -> int:
    """How many chunks to pull for this document and quiz size.

    Rules of thumb (defensible in the report):
    - At least MIN_TOP_K for short docs.
    - About 2 chunks of context per requested question.
    - Up to ~25% of a long document, capped at MAX_TOP_K.
    """
    if num_chunks <= 0:
        return 0
    by_questions = max(MIN_TOP_K, num_questions * 2)
    by_size = max(MIN_TOP_K, math.ceil(num_chunks * 0.25))
    k = max(by_questions, min(by_size, MAX_TOP_K))
    return min(k, num_chunks, MAX_TOP_K)


def allocate_questions(num_questions: int, num_batches: int) -> list[int]:
    """Split N questions as evenly as possible across section batches."""
    if num_batches <= 0:
        return []
    if num_questions <= 0:
        return [0] * num_batches
    base = num_questions // num_batches
    rem = num_questions % num_batches
    # Give the remainder to earlier batches (front of document first).
    return [base + (1 if i < rem else 0) for i in range(num_batches)]


def diversify_indices(
    ranked: list[tuple[int, float]],
    *,
    top_k: int,
    n_chunks: int,
) -> list[tuple[int, float]]:
    """Pick up to top_k hits with a minimum index gap (MMR-style spacing).

    Pure BM25 often returns neighbouring windows from one section. Enforcing
    a gap spreads coverage while still preferring high scores.
    """
    if not ranked or top_k <= 0:
        return []
    if len(ranked) <= top_k and top_k >= n_chunks:
        return ranked[:top_k]

    # Gap grows slowly with document length: at least 1 chunk apart.
    min_gap = max(1, n_chunks // max(top_k * 2, 1))

    selected: list[tuple[int, float]] = []
    selected_idx: list[int] = []

    def far_enough(idx: int) -> bool:
        return all(abs(idx - s) >= min_gap for s in selected_idx)

    for item in ranked:
        if len(selected) >= top_k:
            break
        idx, _score = item
        if far_enough(idx):
            selected.append(item)
            selected_idx.append(idx)

    # If gap was too strict, fill remaining slots by score order.
    if len(selected) < top_k:
        taken = set(selected_idx)
        for item in ranked:
            if len(selected) >= top_k:
                break
            if item[0] not in taken:
                selected.append(item)
                taken.add(item[0])

    selected.sort(key=lambda p: p[0])  # document order for section batching
    return selected


def batch_chunk_groups(
    chunk_indices: list[int],
    chunks: list[str],
    *,
    max_per_batch: int = MAX_CHUNKS_PER_BATCH,
) -> list[list[str]]:
    """Group selected chunks (document order) into batches for generation."""
    if not chunk_indices:
        return []
    ordered = sorted(set(chunk_indices))
    groups: list[list[str]] = []
    for start in range(0, len(ordered), max_per_batch):
        slice_idx = ordered[start : start + max_per_batch]
        groups.append([chunks[i] for i in slice_idx])
    return groups


def plan_retrieval(
    retriever: BM25Retriever,
    *,
    num_questions: int,
    topic: str | None,
    use_rag: bool,
) -> RetrievalPlan:
    """Decide which chunks to use and how to batch them for generation."""
    n = len(retriever.chunks)
    if not use_rag:
        return RetrievalPlan(
            strategy="baseline_none",
            top_k=0,
            chunk_indices=[],
            num_batches=1,
            questions_per_batch=[num_questions],
            num_document_chunks=n,
        )

    top_k = compute_top_k(n, num_questions)

    if topic and topic.strip():
        ranked = retriever.search(topic.strip(), top_k=max(top_k * 3, top_k))
        if ranked:
            hits = diversify_indices(ranked, top_k=top_k, n_chunks=n)
            strategy = "bm25_diversified"
        else:
            hits = retriever.spread_sample(top_k=top_k)
            strategy = "spread_fallback"
    else:
        hits = retriever.spread_sample(top_k=top_k)
        strategy = "spread" if n > top_k else "all_chunks"

    indices = [i for i, _ in hits]
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_indices: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)

    batches = batch_chunk_groups(unique_indices, retriever.chunks)
    if not batches:
        batches = [retriever.chunks[:1]] if retriever.chunks else [[]]
        unique_indices = [0] if retriever.chunks else []

    q_per = allocate_questions(num_questions, len(batches))
    # Drop empty question slots' batches (shouldn't happen unless num_questions=0)
    pairs = [(b, q) for b, q in zip(batches, q_per) if q > 0]
    if pairs:
        batches = [p[0] for p in pairs]
        q_per = [p[1] for p in pairs]

    return RetrievalPlan(
        strategy=strategy,
        top_k=top_k,
        chunk_indices=sorted(unique_indices),
        num_batches=len(batches),
        questions_per_batch=q_per,
        num_document_chunks=n,
    )


def context_batches_from_plan(
    plan: RetrievalPlan,
    chunks: list[str],
) -> list[list[str]]:
    """Materialise text batches from a plan (same grouping as plan_retrieval)."""
    if plan.strategy == "baseline_none":
        return [[]]
    return batch_chunk_groups(plan.chunk_indices, chunks)
