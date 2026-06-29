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

