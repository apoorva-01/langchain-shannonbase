"""Tests for 0.2.0 features: metadata filtering, MMR, and relevance scores.

All run offline against InMemoryStore with a deterministic bag-of-words embedder,
so disjoint vocabularies give orthogonal (dissimilar) vectors on demand.
"""

import hashlib
import re

import pytest
from langchain_core.embeddings import Embeddings

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbeddings(Embeddings):
    dim = 64

    def _embed(self, text):
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _store():
    return ShannonBaseVectorStore(embedding=HashEmbeddings(), store=InMemoryStore())


def test_metadata_filter_restricts_results():
    vs = _store()
    vs.add_texts(
        ["red apple", "red car", "blue sky"],
        metadatas=[{"color": "red"}, {"color": "red"}, {"color": "blue"}],
        ids=["1", "2", "3"],
    )
    hits = vs.similarity_search("anything", k=5, filter={"color": "red"})
    assert {d.id for d in hits} == {"1", "2"}


def test_filter_can_return_nothing():
    vs = _store()
    vs.add_texts(["red apple"], metadatas=[{"color": "red"}], ids=["1"])
    assert vs.similarity_search("apple", k=5, filter={"color": "green"}) == []


def test_mmr_pulls_in_a_diverse_result():
    vs = _store()
    # three identical "apple" docs plus one orthogonal "banana" doc
    vs.add_texts(
        ["apple apple", "apple apple", "apple apple", "banana banana"],
        ids=["a1", "a2", "a3", "b"],
    )
    hits = vs.max_marginal_relevance_search("apple apple", k=2, fetch_k=4, lambda_mult=0.2)
    ids = {d.id for d in hits}
    assert len(hits) == 2
    assert "b" in ids  # diversity should surface the banana over a duplicate apple


def test_relevance_scores_are_normalized():
    vs = _store()
    vs.add_texts(["red apple", "blue sky"], ids=["1", "2"])
    scored = vs.similarity_search_with_relevance_scores("red apple", k=2)
    assert len(scored) == 2
    assert all(0.0 <= score <= 1.0 for _, score in scored)


def test_filter_in_operator():
    vs = _store()
    vs.add_texts(["a", "b", "c"], metadatas=[{"t": "x"}, {"t": "y"}, {"t": "z"}], ids=["1", "2", "3"])
    hits = vs.similarity_search("q", k=5, filter={"t": {"$in": ["x", "z"]}})
    assert {d.id for d in hits} == {"1", "3"}


def test_filter_ne_operator():
    vs = _store()
    vs.add_texts(["a", "b"], metadatas=[{"t": "x"}, {"t": "y"}], ids=["1", "2"])
    hits = vs.similarity_search("q", k=5, filter={"t": {"$ne": "x"}})
    assert {d.id for d in hits} == {"2"}


def test_filter_numeric_range():
    vs = _store()
    vs.add_texts(["a", "b", "c"], metadatas=[{"v": 10}, {"v": 50}, {"v": 100}], ids=["1", "2", "3"])
    hits = vs.similarity_search("q", k=5, filter={"v": {"$gt": 10, "$lte": 50}})
    assert {d.id for d in hits} == {"2"}


def test_unknown_operator_raises():
    vs = _store()
    vs.add_texts(["a"], metadatas=[{"t": "x"}], ids=["1"])
    with pytest.raises(ValueError):
        vs.similarity_search("q", k=5, filter={"t": {"$like": "x"}})


def test_custom_columns_accepted():
    # Column names flow into the Schema without error, even with the in-memory store.
    ShannonBaseVectorStore(embedding=HashEmbeddings(), store=InMemoryStore(),
                           id_column="doc_id", content_column="body", create_table=False)


def test_euclidean_relevance_fn_bounded_and_monotonic():
    fn = ShannonBaseVectorStore._euclidean_relevance_score_fn
    # score = 1 - distance, so smaller score means larger distance.
    r0 = fn(1.0)   # distance 0
    r1 = fn(0.0)   # distance 1
    r2 = fn(-1.0)  # distance 2
    assert r0 == 1.0
    assert 0.0 < r2 < r1 < r0 <= 1.0  # closer -> higher, all in (0, 1]


def test_dot_metric_has_no_relevance_score_fn():
    vs = ShannonBaseVectorStore(embedding=HashEmbeddings(), store=InMemoryStore(), metric="dot")
    vs.add_texts(["a"], ids=["1"])
    with pytest.raises(NotImplementedError):
        vs.similarity_search_with_relevance_scores("a", k=1)
