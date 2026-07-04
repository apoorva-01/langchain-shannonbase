# langchain-shannonbase

A [LangChain](https://python.langchain.com) `VectorStore` for **MySQL 9's native
`VECTOR` type** — so you can do RAG on a database you already run.

Works with **[ShannonBase](https://github.com/Shannon-Data/ShannonBase)** (the
open-source MySQL-for-AI), **self-hosted MySQL 9**, and **MySQL HeatWave** — they
all share the same `VECTOR` / `STRING_TO_VECTOR` / `DISTANCE` surface.

## Why this exists

If your data already lives in MySQL, your options for LangChain vector search were
thin: the only MySQL `VectorStore` is locked to Google Cloud SQL, and ShannonBase's
LangChain integration was on its wishlist but unbuilt. This fills that gap — a
plain, self-hostable adapter that plugs MySQL 9 into the LangChain ecosystem, no
separate vector database required.

## Install

```bash
pip install "langchain-shannonbase[mysql]"
```

## Use

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
)

# It's a normal LangChain vector store — use it directly or as a retriever:
docs = store.similarity_search("return policy?", k=2)
retriever = store.as_retriever(search_kwargs={"k": 3})
```

Because it implements LangChain's `VectorStore` interface, it drops into any
LangChain chain, retriever, or RAG pipeline unchanged.

## How it works

Under the hood it uses MySQL 9's native vector features — no extensions:

```sql
CREATE TABLE documents (
  id VARCHAR(36) PRIMARY KEY,
  content TEXT, metadata JSON,
  embedding VECTOR(1536)
);
-- inserts go through STRING_TO_VECTOR('[...]')
-- search: ORDER BY DISTANCE(embedding, STRING_TO_VECTOR('[...]'), 'COSINE')
```

Similarity search reads back the nearest rows by cosine distance and returns them
as LangChain `Document`s with a score (`1 - distance`).

## API

| Method | Does |
|--------|------|
| `add_texts(texts, metadatas, ids)` | embed + upsert, returns ids |
| `similarity_search(query, k)` | top-k `Document`s |
| `similarity_search_with_score(query, k)` | with cosine similarity scores |
| `similarity_search_by_vector(embedding, k)` | search with a raw vector |
| `delete(ids)` | remove by id |
| `from_texts(texts, embedding, ...)` | build a store in one call |
| `metric=` | `"cosine"` (default), `"dot"`, `"euclidean"` |

## Testing

The core logic is unit-tested offline via an in-memory backend (no database
needed — `pytest`). A live round-trip test runs against a real instance when you
set the connection env vars:

```bash
export SB_HOST=127.0.0.1 SB_USER=root SB_PASSWORD=... SB_DATABASE=test
pytest tests/test_integration.py
```

> Local dev tip: run [ShannonBase](https://github.com/Shannon-Data/ShannonBase) to
> get MySQL-9 vector features without a HeatWave subscription.

## Requirements

- Python 3.9+
- A MySQL-9-compatible database with the `VECTOR` type (ShannonBase, MySQL 9, or
  HeatWave)
- `mysql-connector-python` (installed via the `[mysql]` extra)

## License

MIT © Apoorva Verma
