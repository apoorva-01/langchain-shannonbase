# langchain-shannonbase

[![PyPI](https://img.shields.io/pypi/v/langchain-shannonbase)](https://pypi.org/project/langchain-shannonbase/)
[![CI](https://github.com/apoorva-01/langchain-shannonbase/actions/workflows/ci.yml/badge.svg)](https://github.com/apoorva-01/langchain-shannonbase/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/langchain-shannonbase)](https://pypi.org/project/langchain-shannonbase/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A [LangChain](https://python.langchain.com) `VectorStore` backed by MySQL 9's native `VECTOR` type. If your data already lives in MySQL, you can do retrieval without bolting a separate vector database onto your stack.

It works against three things that share the same `VECTOR` / `STRING_TO_VECTOR` / `DISTANCE` surface:

| Backend | What it is | Good for |
|---|---|---|
| [ShannonBase](https://github.com/Shannon-Data/ShannonBase) | open-source "MySQL for AI" | local dev and self-hosting, no subscription |
| MySQL 9 | vanilla self-hosted MySQL | you already run MySQL |
| MySQL HeatWave | Oracle's managed MySQL | production on OCI |

## Why it exists

Before this, if your data was in MySQL your LangChain options were thin. The one MySQL vector store in the ecosystem is locked to Google Cloud SQL, and ShannonBase's LangChain integration was on their wishlist but nobody had built it. This is the plain, self-hostable version: no cloud lock-in, no extra service to run.

It passes LangChain's [standard vector-store integration suite](https://pypi.org/project/langchain-tests/), so it behaves like any other store you'd drop into a chain.

## When to use this (and when not to)

ShannonBase can also do the whole retrieve-and-generate loop in one SQL call with `sys.ML_RAG`. If you're all-in on ShannonBase and happy in SQL, that's the simpler path and you probably don't need this.

Reach for this package when you're already building in LangChain: you want orchestration in Python, your own embeddings (OpenAI, a local model, whatever), or to plug MySQL into a chain or agent you've already got. Same engine underneath, different front door.

## Install

```bash
pip install "langchain-shannonbase[mysql]"
```

The `[mysql]` extra pulls in the database driver. Leave it off if you only want the offline in-memory backend for tests.

## Quickstart

```python
from langchain_openai import OpenAIEmbeddings
from langchain_shannonbase import ShannonBaseVectorStore

store = ShannonBaseVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    table="documents",
    host="127.0.0.1", port=3306, user="root", password="", database="rag",
)

store.add_texts(
    ["Refunds are accepted within 30 days.", "Free shipping over $50."],
    metadatas=[{"topic": "refunds"}, {"topic": "shipping"}],
    ids=["1", "2"],
)

store.similarity_search("what's the return policy?", k=2)
```

The table is created on the first write, with an `embedding VECTOR(n)` column sized to your embedding model.

For a full doc-in to grounded-answer example, see [`examples/rag.py`](examples/rag.py).

### Filtering, MMR, scores, retriever

```python
# restrict a search to matching metadata
store.similarity_search("policy?", k=2, filter={"topic": "refunds"})

# maximal marginal relevance, for hits that aren't near-duplicates of each other
store.max_marginal_relevance_search("policy?", k=3, fetch_k=20, lambda_mult=0.5)

# cosine similarity, or a normalized [0,1] relevance score, with each hit
store.similarity_search_with_score("return policy?", k=2)
store.similarity_search_with_relevance_scores("return policy?", k=2)

# search with an embedding you already have
store.similarity_search_by_vector(my_vector, k=2)

# fetch or delete specific rows by id
store.get_by_ids(["1"])
store.delete(ids=["2"])

# use it as a retriever in any chain
retriever = store.as_retriever(search_kwargs={"k": 3})
```

## How it works

No extensions, just MySQL 9's built-in vector support:

```sql
CREATE TABLE documents (
  id VARCHAR(36) PRIMARY KEY,
  content TEXT,
  metadata JSON,
  embedding VECTOR(1536)
);
-- inserts go through STRING_TO_VECTOR('[...]')
-- search:  ORDER BY DISTANCE(embedding, STRING_TO_VECTOR('[...]'), 'COSINE') LIMIT k
```

Search returns the nearest rows as LangChain `Document`s, each with a score of `1 - distance`. Cosine is the default; pass `metric="dot"` or `metric="euclidean"` if you'd rather.

## Performance and scale

Search is exact: MySQL 9 (and ShannonBase) run a full `DISTANCE` scan and return the true nearest neighbours, so recall is always 100%. The tradeoff is that latency grows with the row count, since there's no approximate (HNSW-style) vector index in MySQL 9 or ShannonBase yet. In practice this is fine for thousands to low millions of vectors; past that you'd want an ANN index, which I'll add if and when the backends support one.

Connections are pooled (`pool_size` defaults to 5, override it in the constructor), so repeated queries reuse connections instead of reconnecting each time.

There's a latency benchmark in [`bench/benchmark.py`](bench/benchmark.py) if you want numbers for your own instance.

## API

| Method | What it does |
|---|---|
| `add_texts(texts, metadatas, ids)` | embed and upsert, returns the ids |
| `similarity_search(query, k, filter=...)` | top-k `Document`s, optional metadata filter |
| `similarity_search_with_score(query, k)` | same, with similarity scores |
| `similarity_search_with_relevance_scores(query, k)` | with normalized [0,1] scores (cosine) |
| `max_marginal_relevance_search(query, k, fetch_k, lambda_mult)` | diverse results |
| `similarity_search_by_vector(embedding, k)` | search with a raw vector |
| `get_by_ids(ids)` | fetch documents by id |
| `delete(ids)` | remove by id |
| `from_texts(texts, embedding, ...)` | build a populated store in one call |

Metrics: `cosine` (default), `dot`, `euclidean`.

## Testing

The logic is unit-tested offline against an in-memory backend, so you don't need a database to run the suite. That's also how the LangChain standard tests run in CI:

```bash
pip install -e ".[dev]"
pytest
```

There's a live round-trip test too, which runs against a real instance when you give it connection details:

```bash
export SB_HOST=127.0.0.1 SB_USER=root SB_PASSWORD=... SB_DATABASE=test
pytest tests/test_integration.py
```

For local development, [ShannonBase](https://github.com/Shannon-Data/ShannonBase) gives you the MySQL 9 vector features without a HeatWave subscription.

## Roadmap

Next on my list:

- Native async via an async MySQL driver (async already works through LangChain's executor fallback)
- Relevance scores for the `dot` and `euclidean` metrics (cosine is done)
- Optional vector indexing where the backend supports it
- Range and comparison operators in filters (only equality today)

Issues and PRs welcome.

## Requirements

- Python 3.9+
- A MySQL-9-compatible database with the `VECTOR` type (ShannonBase, MySQL 9, or HeatWave)
- `mysql-connector-python` (via the `[mysql]` extra)

## License

MIT
