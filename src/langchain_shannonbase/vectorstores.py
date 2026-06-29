"""ShannonBaseVectorStore — a LangChain VectorStore backed by MySQL 9's native
VECTOR type. Works with ShannonBase, self-hosted MySQL 9, and MySQL HeatWave,
which all share the same VECTOR / STRING_TO_VECTOR / DISTANCE surface.
"""

from __future__ import annotations

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
        return [doc for doc, _ in self.similarity_search_by_vector_with_score(embedding, k)]

    def similarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        rows = self._store.search(embedding, k, self.metric)
        # DISTANCE is smaller-is-closer; expose it as a score (1 - distance).
        return [
            (Document(page_content=r.content, metadata={**r.metadata, "id": r.id}),
             1.0 - r.distance)
            for r in rows
        ]

