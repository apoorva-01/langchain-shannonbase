"""Native async path, offline. InMemoryStore implements the async store surface, so
these exercise the vector store's own async methods (aadd_texts / asimilarity_search
/ aget_by_ids / adelete / ahybrid_search) rather than the executor fallback. The
aiomysql backend mirrors the same surface with real non-blocking I/O.

pytest-asyncio runs these (asyncio_mode = "auto" in pyproject).
"""
from langchain_core.embeddings import Embeddings

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore

_V = {
    "banana smoothie": [1.0, 0.0, 0.0],
    "assorted random notes": [0.7, 0.7, 0.0],
    "python programming tutorial guide": [0.0, 1.0, 0.0],
    "python tutorial": [1.0, 0.0, 0.0],
    "banana": [1.0, 0.0, 0.0],
}


class LookupEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [_V[t] for t in texts]

    def embed_query(self, text):
        return _V[text]


def _store():
    return ShannonBaseVectorStore(embedding=LookupEmbeddings(), store=InMemoryStore())


async def test_async_roundtrip():
    vs = _store()
    ids = await vs.aadd_texts(["banana smoothie", "python programming tutorial guide"],
                              ids=["vec", "kw"])
    assert ids == ["vec", "kw"]
    hits = await vs.asimilarity_search("banana", k=1)
    assert hits[0].id == "vec"


async def test_async_get_and_delete():
    vs = _store()
    await vs.aadd_texts(["banana smoothie"], ids=["vec"])
    got = await vs.aget_by_ids(["vec"])
    assert [d.id for d in got] == ["vec"]
    assert await vs.adelete(["vec"]) is True
    assert await vs.aget_by_ids(["vec"]) == []


async def test_async_hybrid_search_fuses():
    vs = _store()
    await vs.aadd_texts(
        ["banana smoothie", "assorted random notes", "python programming tutorial guide"],
        ids=["vec", "mid", "kw"],
    )
    # same fixture as the sync hybrid test: "kw" is the worst vector match but the
    # best keyword match, so a balanced fusion should lift it to the top.
    top = await vs.ahybrid_search("python tutorial", k=3, vector_weight=0.5)
    assert top[0].id == "kw"


async def test_async_with_score_returns_similarity():
    vs = _store()
    await vs.aadd_texts(["banana smoothie"], ids=["vec"])
    scored = await vs.asimilarity_search_with_score("banana", k=1)
    doc, score = scored[0]
    assert doc.id == "vec"
    assert score == 1.0  # identical vector -> cosine distance 0 -> score 1


def test_aiomysql_store_constructs_and_mirrors_sql():
    # No live database, but this catches import/typo breakage in the aiomysql path
    # and confirms it exposes the async store surface the vector store delegates to.
    from langchain_shannonbase import AsyncMySQLStore, _sql

    store = AsyncMySQLStore(_sql.Schema("docs"), host="127.0.0.1", user="root", db="x")
    for method in ("aupsert", "asearch", "akeyword_search", "aget", "adelete", "aset_clusters"):
        assert callable(getattr(store, method))
