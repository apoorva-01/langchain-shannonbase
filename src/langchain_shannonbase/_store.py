"""Storage backends behind the vector store.

MySQLStore talks to a real ShannonBase / MySQL 9 / HeatWave instance. InMemoryStore
emulates the same behavior in pure Python (cosine over stored vectors) so the vector
store can be fully unit-tested offline, deterministically and without a database.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from . import _filter, _sql


@dataclass
class Row:
    id: str
    content: str
    metadata: dict
    distance: float
    embedding: Optional[List[float]] = None


class Store(Protocol):
    def ensure_table(self, dim: int) -> None: ...
    def upsert(self, rows: List[Tuple[str, str, dict, List[float]]]) -> None: ...
    def search(self, embedding: List[float], k: int, metric: str,
               filter: Optional[dict] = None, with_vector: bool = False,
               clusters: Optional[List[int]] = None) -> List[Row]: ...
    def get(self, ids: List[str]) -> List[Row]: ...
    def delete(self, ids: List[str]) -> None: ...
    def all_embeddings(self) -> List[Tuple[str, List[float]]]: ...
    def write_index(self, centroids: List[List[float]], assignments: dict) -> None: ...
    def set_clusters(self, assignments: dict) -> None: ...
    def read_centroids(self) -> Optional[List[List[float]]]: ...


class MySQLStore:
    """Real backend. Requires: pip install 'langchain-shannonbase[mysql]'."""

    def __init__(self, schema: _sql.Schema, pool_size: int = 5, **connection_kwargs):
        import mysql.connector  # noqa: F401  (lazy; validates the extra is installed)
        self.s = schema
        self._conn_kwargs = connection_kwargs
        self._pool_size = pool_size
        self._pool = None

    def _connect(self):
        # Pooled connections are reused across calls; conn.close() returns them to
        # the pool rather than tearing down a TCP connection each time.
        from mysql.connector import pooling
        if self._pool is None:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="shannonbase", pool_size=self._pool_size, **self._conn_kwargs
            )
        return self._pool.get_connection()

    def ensure_table(self, dim: int) -> None:
        conn = self._connect()
        try:
            conn.cursor().execute(_sql.create_table_sql(self.s, dim))
            conn.commit()
        finally:
            conn.close()

    def upsert(self, rows):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.executemany(
                _sql.insert_sql(self.s),
                [(rid, content, json.dumps(meta), _sql.vector_literal(emb))
                 for rid, content, meta, emb in rows],
            )
            conn.commit()
        finally:
            conn.close()

    def search(self, embedding, k, metric, filter=None, with_vector=False, clusters=None):
        clauses, fparams = _filter.to_sql(filter or {}, self.s.metadata)
        probe = list(clusters) if clusters else []
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                _sql.search_sql(self.s, metric, filter_clauses=clauses,
                                with_vector=with_vector, n_clusters=len(probe)),
                (_sql.vector_literal(embedding), *fparams, *probe, k),
            )
            out = []
            for row in cur.fetchall():
                if with_vector:
                    rid, content, meta, emb, dist = row
                    vec = json.loads(emb) if emb else None
                else:
                    rid, content, meta, dist = row
                    vec = None
                md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
                out.append(Row(rid, content, md, float(dist), vec))
            return out
        finally:
            conn.close()

    def get(self, ids):
        if not ids:
            return []
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_sql.select_by_ids_sql(self.s, len(ids)), tuple(ids))
            out = []
            for rid, content, meta in cur.fetchall():
                md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
                out.append(Row(rid, content, md, 0.0))
            return out
        finally:
            conn.close()

    def delete(self, ids):
        if not ids:
            return
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_sql.delete_sql(self.s, len(ids)), tuple(ids))
            conn.commit()
        finally:
            conn.close()

    def all_embeddings(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_sql.all_embeddings_sql(self.s))
            return [(rid, json.loads(emb)) for rid, emb in cur.fetchall()]
        finally:
            conn.close()

    def write_index(self, centroids, assignments):
        import mysql.connector
        dim = len(centroids[0])
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(_sql.create_ivf_table_sql(self.s, dim))
            cur.execute(_sql.clear_ivf_sql(self.s))
            cur.executemany(_sql.insert_ivf_sql(self.s),
                            [(i, _sql.vector_literal(c)) for i, c in enumerate(centroids)])
            # MySQL has no ADD COLUMN IF NOT EXISTS; a duplicate on rebuild is expected.
            for ddl in (_sql.add_cluster_column_sql(self.s), _sql.add_cluster_index_sql(self.s)):
                try:
                    cur.execute(ddl)
                except mysql.connector.Error:
                    pass
            self._update_clusters(cur, assignments)
            conn.commit()
        finally:
            conn.close()

    def set_clusters(self, assignments):
        if not assignments:
            return
        conn = self._connect()
        try:
            cur = conn.cursor()
            self._update_clusters(cur, assignments)
            conn.commit()
        finally:
            conn.close()

    def _update_clusters(self, cur, assignments):
        cur.executemany(_sql.update_cluster_sql(self.s),
                        [(cid, rid) for rid, cid in assignments.items()])

    def read_centroids(self):
        import mysql.connector
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(_sql.select_centroids_sql(self.s))
            except mysql.connector.Error:
                return None  # index has not been built yet
            rows = cur.fetchall()
            return [json.loads(vec) for _, vec in rows] if rows else None
        finally:
            conn.close()


class InMemoryStore:
    """Deterministic offline backend that mirrors MySQLStore's cosine behavior."""

    def __init__(self):
        self._rows: dict[str, Tuple[str, dict, List[float]]] = {}
        self._centroids: Optional[List[List[float]]] = None
        self._cluster: dict[str, int] = {}

    def ensure_table(self, dim: int) -> None:
        pass

    def upsert(self, rows):
        for rid, content, meta, emb in rows:
            self._rows[rid] = (content, dict(meta or {}), list(emb))

    def search(self, embedding, k, metric, filter=None, with_vector=False, clusters=None):
        probe = set(clusters) if clusters is not None else None
        scored = []
        for rid, (content, meta, emb) in self._rows.items():
            if probe is not None and self._cluster.get(rid) not in probe:
                continue
            if filter and not _filter.matches(filter, meta):
                continue
            scored.append(Row(rid, content, meta, _cosine_distance(embedding, emb),
                              list(emb) if with_vector else None))
        scored.sort(key=lambda r: r.distance)
        return scored[:k]

    def all_embeddings(self):
        return [(rid, list(emb)) for rid, (_, _, emb) in self._rows.items()]

    def write_index(self, centroids, assignments):
        self._centroids = [list(c) for c in centroids]
        self._cluster = dict(assignments)

    def set_clusters(self, assignments):
        self._cluster.update(assignments)

    def read_centroids(self):
        return self._centroids

    def get(self, ids):
        out = []
        for i in ids:
            row = self._rows.get(i)
            if row is not None:
                content, meta, _ = row
                out.append(Row(i, content, dict(meta), 0.0))
        return out

    def delete(self, ids):
        for i in ids:
            self._rows.pop(i, None)
            self._cluster.pop(i, None)


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)
