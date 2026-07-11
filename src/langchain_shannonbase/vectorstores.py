"""ShannonBaseVectorStore — a LangChain VectorStore backed by MySQL 9's native
VECTOR type. Works with ShannonBase, self-hosted MySQL 9, and MySQL HeatWave,
which all share the same VECTOR / STRING_TO_VECTOR / DISTANCE surface.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ._store import InMemoryStore, MySQLStore, Store


class ShannonBaseVectorStore(VectorStore):
    """Store and query embeddings in a MySQL-9-compatible database.

    Typical use points it at a real database:

        from langchain_shannonbase import ShannonBaseVectorStore
        store = ShannonBaseVectorStore(
            embedding=my_embeddings,
            table="documents",
            host="127.0.0.1", user="root", password="...", database="rag",
        )
        store.add_texts(["hello world"], metadatas=[{"src": "demo"}])
        docs = store.similarity_search("greeting", k=3)

    Pass `store=InMemoryStore()` instead of connection kwargs for offline tests.
    """

    def __init__(
        self,
        embedding: Embeddings,
        table: str = "langchain_vectors",
        metric: str = "cosine",
        store: Optional[Store] = None,
        **connection_kwargs: Any,
    ):
        self._embedding = embedding
        self.table = table
        self.metric = metric
        self._store: Store = store if store is not None else MySQLStore(table, **connection_kwargs)
        self._dim: Optional[int] = None

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            self._store.ensure_table(dim)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if not texts:
            return []
        vectors = self._embedding.embed_documents(texts)
        self._ensure_dim(len(vectors[0]))
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        rows: List[Tuple[str, str, dict, List[float]]] = [
            (ids[i], texts[i], metadatas[i], vectors[i]) for i in range(len(texts))
        ]
        self._store.upsert(rows)
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(vector, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(embedding, k, **kwargs)]

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        rows = self._store.search(embedding, k, self.metric, filter=filter)
        # DISTANCE is smaller-is-closer; expose it as a score (1 - distance).
        return [
            (Document(id=r.id, page_content=r.content, metadata=dict(r.metadata)),
             1.0 - r.distance)
            for r in rows
        ]

    def get_by_ids(self, ids: Iterable[str]) -> List[Document]:
        ids = list(ids)
        found = {r.id: r for r in self._store.get(ids)}
        return [
            Document(id=i, page_content=found[i].content, metadata=dict(found[i].metadata))
            for i in ids
            if i in found
        ]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        rows = self._store.search(embedding, fetch_k, self.metric, filter=filter, with_vector=True)
        if not rows:
            return []
        picks = _mmr(embedding, [r.embedding for r in rows], k, lambda_mult)
        return [
            Document(id=rows[i].id, page_content=rows[i].content, metadata=dict(rows[i].metadata))
            for i in picks
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, filter=filter
        )

    @staticmethod
    def _cosine_relevance_score_fn(score: float) -> float:
        # similarity_search_with_score already returns cosine similarity (1 - distance);
        # clamp to [0, 1] for the relevance-score API.
        return max(0.0, min(1.0, score))

    def _select_relevance_score_fn(self):
        if self.metric == "cosine":
            return self._cosine_relevance_score_fn
        return super()._select_relevance_score_fn()

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        self._store.delete(ids)
        return True

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        table: str = "langchain_vectors",
        metric: str = "cosine",
        store: Optional[Store] = None,
        ids: Optional[List[str]] = None,
        **connection_kwargs: Any,
    ) -> "ShannonBaseVectorStore":
        vs = cls(embedding=embedding, table=table, metric=metric, store=store, **connection_kwargs)
        vs.add_texts(texts, metadatas=metadatas, ids=ids)
        return vs


def _cos_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _mmr(query: List[float], candidates: List[List[float]], k: int, lambda_mult: float) -> List[int]:
    """Maximal marginal relevance: pick k candidates that are close to the query
    but not to each other. Returns the chosen candidate indices."""
    if not candidates:
        return []
    k = min(k, len(candidates))
    to_query = [_cos_sim(query, c) for c in candidates]
    selected: List[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        best_i, best_score = None, None
        for i in remaining:
            redundancy = max((_cos_sim(candidates[i], candidates[j]) for j in selected), default=0.0)
            score = lambda_mult * to_query[i] - (1.0 - lambda_mult) * redundancy
            if best_score is None or score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected
