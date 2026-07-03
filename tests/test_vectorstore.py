"""End-to-end tests of ShannonBaseVectorStore against the offline InMemoryStore.

A deterministic bag-of-words embedder stands in for a real embedding model so the
semantics (paraphrases rank closer than unrelated text) are reproducible offline.
"""

import hashlib
import re

from langchain_core.embeddings import Embeddings

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbeddings(Embeddings):
    dim = 64

    def _embed(self, text: str):
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


def test_add_and_search_returns_relevant_doc():
    vs = _store()
    vs.add_texts(
        ["the cat sat on the mat", "quarterly financial report", "a dog in the park"],
        metadatas=[{"src": "a"}, {"src": "b"}, {"src": "c"}],
    )
    docs = vs.similarity_search("cat on a mat", k=1)
    assert len(docs) == 1
    assert "cat" in docs[0].page_content
    assert docs[0].metadata["src"] == "a"


def test_add_texts_returns_ids_and_respects_custom_ids():
    vs = _store()
    ids = vs.add_texts(["hello"], ids=["fixed-id"])
    assert ids == ["fixed-id"]


def test_similarity_search_with_score_orders_by_similarity():
    vs = _store()
    vs.add_texts(["reset my password", "cancel my subscription"])
    results = vs.similarity_search_with_score("how do I reset my password", k=2)
    assert results[0][0].page_content == "reset my password"
    assert results[0][1] >= results[1][1]  # higher score first


def test_k_limits_results():
    vs = _store()
    vs.add_texts([f"doc number {i}" for i in range(10)])
    assert len(vs.similarity_search("doc", k=3)) == 3


def test_delete_removes_document():
    vs = _store()
    vs.add_texts(["keep me", "delete me"], ids=["keep", "del"])
    assert vs.delete(["del"]) is True
    remaining = vs.similarity_search("me", k=10)
    assert all(d.metadata["id"] != "del" for d in remaining)


def test_from_texts_classmethod():
    vs = ShannonBaseVectorStore.from_texts(
        ["alpha", "beta"], embedding=HashEmbeddings(), store=InMemoryStore()
    )
    assert len(vs.similarity_search("alpha", k=2)) == 2
