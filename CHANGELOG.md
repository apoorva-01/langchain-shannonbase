# Changelog

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
