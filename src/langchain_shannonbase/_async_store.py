"""Async backend using aiomysql. It's a query-for-query mirror of MySQLStore that
reuses the same `_sql` builders, so the only difference is non-blocking I/O: the
SQL is identical, which is what keeps the two paths in step.

Requires the async extra: pip install 'langchain-shannonbase[async]'. MySQL 9
defaults to caching_sha2_password, so aiomysql needs `cryptography` installed
(it's pulled in by the extra).
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from . import _filter, _sql
from ._store import Row


class AsyncMySQLStore:
    def __init__(self, schema: _sql.Schema, pool_minsize: int = 1,
                 pool_maxsize: int = 5, **connection_kwargs):
        import aiomysql  # noqa: F401  (lazy; validates the extra is installed)
        self.s = schema
        self._conn_kwargs = connection_kwargs
        self._minsize = pool_minsize
        self._maxsize = pool_maxsize
        self._pool = None

    async def _pool_(self):
        import aiomysql
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                minsize=self._minsize, maxsize=self._maxsize,
                autocommit=True, **self._conn_kwargs,
            )
        return self._pool

    async def _fetch(self, sql: str, params=()):
        pool = await self._pool_()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                return await cur.fetchall()

    async def _run(self, sql: str, params=()):
        pool = await self._pool_()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)

    async def aensure_table(self, dim: int) -> None:
        await self._run(_sql.create_table_sql(self.s, dim))

    async def aensure_fulltext_index(self) -> None:
        import aiomysql
        try:
            await self._run(_sql.add_fulltext_index_sql(self.s))
        except aiomysql.Error:
            pass

    async def aupsert(self, rows) -> None:
        pool = await self._pool_()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    _sql.insert_sql(self.s),
                    [(rid, content, json.dumps(meta), _sql.vector_literal(emb))
                     for rid, content, meta, emb in rows],
                )

    async def asearch(self, embedding, k, metric, filter=None,
                      with_vector=False, clusters=None) -> List[Row]:
        clauses, fparams = _filter.to_sql(filter or {}, self.s.metadata)
        probe = list(clusters) if clusters else []
        rows = await self._fetch(
            _sql.search_sql(self.s, metric, filter_clauses=clauses,
                            with_vector=with_vector, n_clusters=len(probe)),
            (_sql.vector_literal(embedding), *fparams, *probe, k),
        )
        out = []
        for row in rows:
            if with_vector:
                rid, content, meta, emb, dist = row
                vec = json.loads(emb) if emb else None
            else:
                rid, content, meta, dist = row
                vec = None
            md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            out.append(Row(rid, content, md, float(dist), vec))
        return out

    async def akeyword_search(self, text, k, filter=None) -> List[Row]:
        clauses, fparams = _filter.to_sql(filter or {}, self.s.metadata)
        rows = await self._fetch(
            _sql.keyword_search_sql(self.s, filter_clauses=clauses),
            (text, *fparams, text, k),
        )
        out = []
        for rid, content, meta, score in rows:
            md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            out.append(Row(rid, content, md, -float(score)))
        return out

    async def aget(self, ids) -> List[Row]:
        if not ids:
            return []
        rows = await self._fetch(_sql.select_by_ids_sql(self.s, len(ids)), tuple(ids))
        out = []
        for rid, content, meta in rows:
            md = meta if isinstance(meta, dict) else json.loads(meta or "{}")
            out.append(Row(rid, content, md, 0.0))
        return out

    async def adelete(self, ids) -> None:
        if not ids:
            return
        await self._run(_sql.delete_sql(self.s, len(ids)), tuple(ids))

    async def aset_clusters(self, assignments) -> None:
        if not assignments:
            return
        pool = await self._pool_()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    _sql.update_cluster_sql(self.s),
                    [(cid, rid) for rid, cid in assignments.items()],
                )

    async def aclose(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
