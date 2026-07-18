# Changelog

## 0.6.0
- Hybrid search: `hybrid_search(query, k, vector_weight=...)` blends vector
  similarity with MySQL `FULLTEXT` keyword matching using reciprocal rank fusion,
  so a query that's strong on keywords but weak on embeddings (or vice versa) still
  surfaces the right rows. `vector_weight` (0..1) tilts the blend.
- New tables get a `FULLTEXT` index on the content column automatically. For a
  table created by an earlier version, call `ensure_fulltext_index()` once.
- Relevance scores for the `euclidean` metric, bounded to (0, 1]
  (`similarity_search_with_relevance_scores`). `dot` still has no honest [0, 1]
  mapping (its `DISTANCE` semantics aren't documented) and is left unsupported.

## 0.5.0
- Metadata filter operators: `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`,
  `$lte` (a bare `{"key": value}` is still equality).
- Custom schema: `id_column` / `content_column` / `metadata_column` /
  `embedding_column` and `create_table=False` to use an existing table.
- IVF index stays current on insert: rows added after `build_index` are assigned
  to their nearest centroid, so indexed search finds them without a rebuild.

## 0.4.0
- Approximate IVF index via `build_index(n_lists, nprobe)`: k-means centroids plus
  an indexed `cluster` column, so a search scans only the `nprobe` nearest lists
  instead of the whole table (recall rises with `nprobe`). Exact search stays the
  default; the index is opt-in.
- Offline recall tests measure the tradeoff (probing every list matches exact).

## 0.3.0
- Connection pooling in the MySQL backend (`pool_size`, defaults to 5), so
  repeated queries reuse connections instead of reconnecting each call.
- Ships a `py.typed` marker so downstream users get the type hints; mypy runs in CI.
- Copy-paste RAG example (`examples/rag.py`) and a latency benchmark (`bench/benchmark.py`).
- README: honest note on exact search and how it scales.

## 0.2.0
- Metadata filtering on search: `similarity_search(query, k, filter={"topic": "x"})`,
  translated to a `metadata->>'$.key'` WHERE clause (values always bound, keys validated).
- Maximal marginal relevance search (`max_marginal_relevance_search` /
  `_by_vector`) for diverse results, implemented in pure Python (no numpy).
- Cosine relevance scores via `_select_relevance_score_fn`, so
  `similarity_search_with_relevance_scores` returns values in [0, 1].

## 0.1.0
- Initial release.
- `ShannonBaseVectorStore` implementing LangChain's VectorStore interface on
  MySQL 9's native VECTOR type (ShannonBase / MySQL 9 / HeatWave).
- add_texts, similarity_search(_with_score / _by_vector), get_by_ids, delete, from_texts.
- cosine / dot / euclidean metrics.
- Offline InMemoryStore for deterministic tests; gated live integration test.
- Passes LangChain's standard VectorStore integration suite (langchain-tests),
  run offline against the in-memory backend.
