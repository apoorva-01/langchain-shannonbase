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
# filter by metadata: equality, membership ($in/$nin), or comparison ($gt/$gte/$lt/$lte/$ne)
store.similarity_search("policy?", k=2, filter={"topic": "refunds"})
store.similarity_search("policy?", k=2, filter={"topic": {"$in": ["refunds", "returns"]}})
store.similarity_search("policy?", k=2, filter={"views": {"$gte": 100}})

# hybrid: blend vector similarity with keyword (FULLTEXT) matching
store.hybrid_search("O'Brien refund policy", k=3, vector_weight=0.5)

# maximal marginal relevance, for hits that aren't near-duplicates of each other
store.max_marginal_relevance_search("policy?", k=3, fetch_k=20, lambda_mult=0.5)

# cosine similarity, or a normalized [0,1] relevance score, with each hit
store.similarity_search_with_score("return policy?", k=2)
store.similarity_search_with_relevance_scores("return policy?", k=2)

# search with an embedding you already have
store.similarity_search_by_vector(my_vector, k=2)

# fetch or delete specific rows by id, or delete everything matching a filter
store.get_by_ids(["1"])
store.delete(ids=["2"])
store.delete(filter={"topic": "shipping"})

# use it as a retriever in any chain
retriever = store.as_retriever(search_kwargs={"k": 3})
```

## Hybrid search

Vector search is great at meaning and bad at exact tokens: a product SKU, an error
code, a surname like `O'Brien`. Keyword search is the opposite. `hybrid_search`
runs both and fuses the two rankings, so you get semantic recall without losing the
literal matches.

```python
store.hybrid_search("error E4021 on checkout", k=5, vector_weight=0.5)
```

It uses [reciprocal rank fusion](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf):
each retriever returns its top `fetch_k`, and a document's score is the weighted sum
of `1 / (rank + k)` across the two lists. Fusing on *rank* means the two very
different score scales (cosine distance vs. a FULLTEXT relevance score) never have
to be reconciled. `vector_weight` runs 0 (pure keyword) to 1 (pure vector); 0.5 is
a sensible start.

The keyword half needs a MySQL `FULLTEXT` index on the content column. Tables
created by this package (0.6.0+) get one automatically. For a table created by an
earlier version, or your own table, add it once:

```python
store.ensure_fulltext_index()
```

Two InnoDB FULLTEXT quirks worth knowing, because they're easy to mistake for bugs:
the default minimum token length is 3 (`innodb_ft_min_token_size`), so one- and
two-character terms are ignored, and common words on the stopword list don't match.
Both are server settings, not something this package controls.

## Async

If you're in an async app, use the `a`-prefixed methods and they'll do non-blocking
I/O through [aiomysql](https://github.com/aio-libs/aiomysql) instead of tying up the
event loop:

```python
from langchain_shannonbase import ShannonBaseVectorStore

store = ShannonBaseVectorStore(embedding=embeddings, table="documents",
                               host="127.0.0.1", user="root", password="", database="rag")
await store.aadd_texts(["hello world"])
docs = await store.asimilarity_search("greeting", k=3)
hits = await store.ahybrid_search("error E4021", k=5)
```

Install the driver with the async extra:

```bash
pip install "langchain-shannonbase[async]"
```

MySQL 9 defaults to `caching_sha2_password`, which aiomysql needs `cryptography`
for; the extra pulls it in. `aadd_texts`, `asimilarity_search[_with_score]`,
`aget_by_ids`, `adelete`, and `ahybrid_search` are wired to the async driver. Any
async method you don't see here still works through LangChain's thread-pool
fallback, and so does everything if you haven't installed the async extra.

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

Search returns the nearest rows as LangChain `Document`s, each with a score of `1 - distance`. Cosine is the default; pass `metric="dot"` or `metric="euclidean"` if you'd rather. `similarity_search_with_relevance_scores` normalizes to [0, 1] for all three metrics. `dot` uses the fact that ShannonBase's `DISTANCE(...,'DOT')` is the negated inner product, so on normalized embeddings (what most models produce) the dot score matches cosine; larger inner products from non-unit vectors clamp to 1.

## Performance and scale

By default search is exact: a full `DISTANCE` scan that returns the true nearest neighbours, so recall is 100%. That's fine for thousands to low millions of vectors.

Past that, build an approximate **IVF index** so a search only scans a fraction of the table:

```python
store.add_texts(my_docs)           # load your data first
store.build_index(n_lists=1000)    # k-means centroids + an indexed cluster column

store.similarity_search("query", k=5, nprobe=10)   # scans ~ nprobe / n_lists of the rows
```

It's k-means clustering plus an indexed `cluster` column, the same idea as pgvector's `IVFFlat`, done in application logic because MySQL 9 and ShannonBase don't have a native ANN index yet. Recall is approximate and rises with `nprobe`. On clustered data the trade is steep in your favour: in the offline tests, probing 1 of 8 lists keeps recall@10 at 1.0 while scanning ~12% of rows. Real recall depends on your data, so measure with `bench/benchmark.py` on your own set.

Rows added after `build_index` are assigned to their nearest centroid automatically, so the index stays correct as you keep writing. Rebuild periodically (call `build_index` again) to re-centre the clusters as the data grows.

### Native indexes and HeatWave

People ask why there's an app-side index at all, so: ShannonBase and self-hosted
MySQL 9 do a brute-force `DISTANCE` scan with no built-in ANN index (I read the
ShannonBase source to be sure). That's exactly why the IVF index above exists.

MySQL HeatWave is the exception. As of HeatWave 9.5.0 it builds an HNSW vector index
automatically in the background for frequently-queried `VECTOR` columns, and applies
it transparently to `ORDER BY DISTANCE(...) LIMIT k` queries. That's the exact shape
this library emits, so on HeatWave your existing `similarity_search` picks up the
native index with no code change and no DDL. **Don't call `build_index` on HeatWave**,
the IVF index is only for backends that lack a native one.

Recall vs. latency on HeatWave is tuned server-side with the `rapid_hnsw_ef_search`
session variable (higher is more accurate, slower); it's a HeatWave setting, not
something this library manages.

Connections are pooled (`pool_size` defaults to 5, override it in the constructor), so repeated queries reuse connections instead of reconnecting each time.

There's a latency benchmark in [`bench/benchmark.py`](bench/benchmark.py) if you want numbers for your own instance.

## API

| Method | What it does |
|---|---|
| `add_texts(texts, metadatas, ids)` | embed and upsert, returns the ids |
| `hybrid_search(query, k, vector_weight=...)` | vector + FULLTEXT keyword, fused by rank |
| `similarity_search(query, k, filter=...)` | top-k `Document`s, optional metadata filter |
| `similarity_search_with_score(query, k)` | same, with similarity scores |
| `similarity_search_with_relevance_scores(query, k)` | with normalized [0,1] scores (cosine) |
| `max_marginal_relevance_search(query, k, fetch_k, lambda_mult)` | diverse results |
| `similarity_search_by_vector(embedding, k)` | search with a raw vector |
| `get_by_ids(ids)` | fetch documents by id |
| `delete(ids)` | remove by id |
| `from_texts(texts, embedding, ...)` | build a populated store in one call |
| `build_index(n_lists, nprobe)` | build an approximate IVF index for large tables |

Metrics: `cosine` (default), `dot`, `euclidean`.

## Custom schema

By default the store creates and owns its table. To point it at an existing table, or to use your own column names, pass them in and turn off table creation:

```python
store = ShannonBaseVectorStore(
    embedding=embeddings,
    table="my_docs",
    id_column="doc_id",
    content_column="body",
    metadata_column="meta",
    embedding_column="vec",
    create_table=False,          # don't CREATE TABLE; use what's already there
    host="127.0.0.1", user="root", password="", database="app",
)
```

Column and table names are validated as SQL identifiers.

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

- LlamaIndex sibling package, so the same MySQL backend works outside LangChain
- More `bench/` coverage: recall-vs-nprobe curves you can reproduce on your own data

On native ANN indexes: HeatWave already builds one automatically and this library's
queries use it transparently (see above). ShannonBase and self-hosted MySQL don't
have one, so the IVF index via `build_index` is the answer there until they do.

Done recently: native async (aiomysql), relevance scores for all three metrics,
hybrid search (vector + `FULLTEXT`), an approximate IVF index, custom schemas, and
metadata filter operators.

Issues and PRs welcome.

## Requirements

- Python 3.9+
- A MySQL-9-compatible database with the `VECTOR` type (ShannonBase, MySQL 9, or HeatWave)
- `mysql-connector-python` (via the `[mysql]` extra)

## License

MIT
