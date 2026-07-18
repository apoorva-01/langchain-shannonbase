"""Hybrid search, offline. The point of the fixture is that the vector-best and
keyword-best documents are *different*, so a passing test actually exercises the
reciprocal-rank fusion rather than agreeing by accident.
"""
from langchain_core.embeddings import Embeddings

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore

# text -> vector. Query "python tutorial" points along [1,0,0].
_VEC = {
    "banana smoothie": [1.0, 0.0, 0.0],          # vector-best, no keyword overlap
    "assorted random notes": [0.7, 0.7, 0.0],     # middle on vector, no overlap
    "python programming tutorial guide": [0.0, 1.0, 0.0],  # vector-worst, keyword-best
    "python tutorial": [1.0, 0.0, 0.0],           # the query
}


class LookupEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [_VEC[t] for t in texts]

    def embed_query(self, text):
        return _VEC[text]


def _store():
    vs = ShannonBaseVectorStore(embedding=LookupEmbeddings(), store=InMemoryStore())
    vs.add_texts(
        ["banana smoothie", "assorted random notes", "python programming tutorial guide"],
        metadatas=[{"lang": "en"}, {"lang": "en"}, {"lang": "fr"}],
        ids=["vec", "mid", "kw"],
    )
    return vs


def test_pure_vector_weight_ranks_by_vector():
    vs = _store()
    top = vs.hybrid_search("python tutorial", k=3, vector_weight=1.0)
    assert top[0].id == "vec"  # nearest vector wins, keyword ignored


def test_pure_keyword_weight_ranks_by_keyword():
    vs = _store()
    top = vs.hybrid_search("python tutorial", k=3, vector_weight=0.0)
    assert top[0].id == "kw"  # only doc that matches the words


def test_fusion_surfaces_keyword_hit_a_vector_search_would_bury():
    # "kw" is the worst vector match (rank 2) but the best keyword match. With a
    # balanced blend it should come out on top, which pure vector search never does.
    vs = _store()
    fused = vs.hybrid_search("python tutorial", k=3, vector_weight=0.5)
    assert fused[0].id == "kw"
    assert {d.id for d in fused} == {"vec", "mid", "kw"}


def test_hybrid_respects_metadata_filter():
    vs = _store()
    # kw is lang=fr; filtering to en drops it from both retrievers.
    fused = vs.hybrid_search("python tutorial", k=3, filter={"lang": "en"})
    assert "kw" not in {d.id for d in fused}


def test_bad_vector_weight_raises():
    import pytest

    vs = _store()
    with pytest.raises(ValueError):
        vs.hybrid_search("python tutorial", vector_weight=1.5)
