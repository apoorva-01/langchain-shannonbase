"""Storage backends behind the vector store.

MySQLStore talks to a real ShannonBase / MySQL 9 / HeatWave instance. InMemoryStore
emulates the same behavior in pure Python (cosine over stored vectors) so the
vector store can be fully unit-tested offline — same pattern that keeps the tests
deterministic and CI-friendly without provisioning a database.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from . import _sql


@dataclass
class Row:
    id: str
    content: str
    metadata: dict
    distance: float


class Store(Protocol):
    def ensure_table(self, dim: int) -> None: ...
    def upsert(self, rows: List[Tuple[str, str, dict, List[float]]]) -> None: ...
    def search(self, embedding: List[float], k: int, metric: str) -> List[Row]: ...
    def delete(self, ids: List[str]) -> None: ...


class MySQLStore:
    """Real backend. Requires: pip install 'langchain-shannonbase[mysql]'."""

    def __init__(self, table: str, **connection_kwargs):
        import mysql.connector  # noqa: F401  (lazy; validates the extra is installed)
        self.table = table
        self._conn_kwargs = connection_kwargs

    def _connect(self):
        import mysql.connector
        return mysql.connector.connect(**self._conn_kwargs)

    def ensure_table(self, dim: int) -> None:
        conn = self._connect()
        try:
            conn.cursor().execute(_sql.create_table_sql(self.table, dim))
            conn.commit()
        finally:
            conn.close()

    def upsert(self, rows):
        conn = self._connect()
        try:
            cur = conn.cursor()
            stmt = _sql.insert_sql(self.table)
            cur.executemany(
                stmt,
                [(rid, content, json.dumps(meta), _sql.vector_literal(emb))
                 for rid, content, meta, emb in rows],
            )
            conn.commit()
        finally:
            conn.close()

    def search(self, embedding, k, metric):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_sql.search_sql(self.table, metric),
                        (_sql.vector_literal(embedding), k))
            out = []
            for rid, content, meta, dist in cur.fetchall():
                md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
                out.append(Row(rid, content, md, float(dist)))
            return out
        finally:
            conn.close()

    def delete(self, ids):
        if not ids:
            return
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_sql.delete_sql(self.table, len(ids)), tuple(ids))
            conn.commit()
        finally:
            conn.close()


