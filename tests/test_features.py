"""Tests for 0.2.0 features: metadata filtering, MMR, and relevance scores.

All run offline against InMemoryStore with a deterministic bag-of-words embedder,
so disjoint vocabularies give orthogonal (dissimilar) vectors on demand.
"""

import hashlib
import re

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
