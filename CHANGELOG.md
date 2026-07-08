# Changelog

## 0.1.0
- Initial release.
- `ShannonBaseVectorStore` implementing LangChain's VectorStore interface on
  MySQL 9's native VECTOR type (ShannonBase / MySQL 9 / HeatWave).
- add_texts, similarity_search(_with_score / _by_vector), get_by_ids, delete, from_texts.
- cosine / dot / euclidean metrics.
- Offline InMemoryStore for deterministic tests; gated live integration test.
- Passes LangChain's standard VectorStore integration suite (langchain-tests),
  run offline against the in-memory backend.
