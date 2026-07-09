"""Tests for scaled retrieval and section batching (no API key)."""

from app.coverage import (
    MAX_TOP_K,
    allocate_questions,
    batch_chunk_groups,
    compute_top_k,
    diversify_indices,
    plan_retrieval,
)
from app.retriever import BM25Retriever


def _retriever(n: int) -> BM25Retriever:
    # Distinct tokens per chunk so BM25 can discriminate.
    chunks = [f"chunk {i} topic{i % 5} content about subject number {i}" for i in range(n)]
    r = BM25Retriever()
    r.index(chunks)
    return r


def test_compute_top_k_scales_with_questions_and_size():
    assert compute_top_k(3, 5) == 3  # cannot exceed doc
    assert compute_top_k(100, 5) >= 6
    assert compute_top_k(100, 10) >= 12
    assert compute_top_k(100, 15) <= MAX_TOP_K


def test_allocate_questions_even():
    assert allocate_questions(5, 2) == [3, 2]
    assert allocate_questions(6, 3) == [2, 2, 2]
    assert allocate_questions(5, 1) == [5]
    assert sum(allocate_questions(15, 4)) == 15


def test_spread_sample_unique_indices():
    r = _retriever(40)
    hits = r.spread_sample(top_k=12)
    indices = [i for i, _ in hits]
    assert len(indices) == 12
    assert len(set(indices)) == 12
    assert indices[0] == 0
    assert indices[-1] == 39


def test_diversify_spreads_hits():
    # Fake ranked list: all mass on neighbouring chunks 10-20.
    ranked = [(i, 100 - i) for i in range(10, 25)]
    picked = diversify_indices(ranked, top_k=5, n_chunks=50)
    idxs = [i for i, _ in picked]
    assert len(idxs) == 5
    # Spaced: not five consecutive indices
    span = max(idxs) - min(idxs)
    assert span >= 4


def test_batch_chunk_groups():
    chunks = [f"c{i}" for i in range(20)]
    groups = batch_chunk_groups([0, 1, 2, 10, 11, 12, 18, 19], chunks, max_per_batch=3)
    assert len(groups) == 3
    assert groups[0] == ["c0", "c1", "c2"]


def test_plan_spread_for_long_doc():
    r = _retriever(50)
    plan = plan_retrieval(r, num_questions=10, topic=None, use_rag=True)
    assert plan.strategy in ("spread", "all_chunks")
    assert plan.top_k >= 6
    assert len(plan.chunk_indices) == plan.top_k or len(plan.chunk_indices) <= plan.top_k
    assert plan.num_batches >= 1
    assert sum(plan.questions_per_batch) == 10


def test_plan_topic_uses_bm25_diversified():
    r = _retriever(40)
    plan = plan_retrieval(r, num_questions=5, topic="topic2", use_rag=True)
    assert plan.strategy in ("bm25_diversified", "spread_fallback")
    assert len(plan.chunk_indices) > 0


def test_plan_baseline_empty_context():
    r = _retriever(10)
    plan = plan_retrieval(r, num_questions=5, topic=None, use_rag=False)
    assert plan.strategy == "baseline_none"
    assert plan.chunk_indices == []
    assert plan.questions_per_batch == [5]
