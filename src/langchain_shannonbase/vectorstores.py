"""ShannonBaseVectorStore — a LangChain VectorStore backed by MySQL 9's native
VECTOR type. Works with ShannonBase, self-hosted MySQL 9, and MySQL HeatWave,
which all share the same VECTOR / STRING_TO_VECTOR / DISTANCE surface.
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from . import _ivf, _sql
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
        id_column: str = "id",
        content_column: str = "content",
        metadata_column: str = "metadata",
        embedding_column: str = "embedding",
        cluster_column: str = "cluster",
        create_table: bool = True,
        **connection_kwargs: Any,
    ):
        self._embedding = embedding
        self.table = table
        self.metric = metric
        self._create_table = create_table
        schema = _sql.Schema(
            table=table, id=id_column, content=content_column, metadata=metadata_column,
            embedding=embedding_column, cluster=cluster_column,
        )
        self._store: Store = store if store is not None else MySQLStore(schema, **connection_kwargs)
        self._dim: Optional[int] = None
        self._nprobe = 8
        self._centroids_loaded = False
        self._centroids: Optional[List[List[float]]] = None

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def _ensure_dim(self, dim: int) -> None:
        if self._dim is None:
            self._dim = dim
            if self._create_table:
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
        # Keep the IVF index current: assign new rows to their nearest centroid so
        # an indexed search finds them without a full rebuild.
        centroids = self._current_centroids()
        if centroids:
            self._store.set_clusters(
                {ids[i]: _ivf.nearest(vectors[i], centroids) for i in range(len(texts))}
            )
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
        clusters = self._probe(embedding, kwargs.get("nprobe"))
        rows = self._store.search(embedding, k, self.metric, filter=filter, clusters=clusters)
        # DISTANCE is smaller-is-closer; expose it as a score (1 - distance).
        return [
            (Document(id=r.id, page_content=r.content, metadata=dict(r.metadata)),
             1.0 - r.distance)
            for r in rows
        ]

    def hybrid_search(
        self, query: str, k: int = 4, fetch_k: int = 20,
        filter: Optional[dict] = None, vector_weight: float = 0.5, **kwargs: Any,
    ) -> List[Document]:
        return [
            doc for doc, _ in self.hybrid_search_with_score(
                query, k, fetch_k, filter=filter, vector_weight=vector_weight, **kwargs
            )
        ]

    def hybrid_search_with_score(
        self, query: str, k: int = 4, fetch_k: int = 20,
        filter: Optional[dict] = None, vector_weight: float = 0.5,
        rrf_k: int = 60, **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Combine vector similarity and keyword (FULLTEXT) matching.

        Both retrievers return their top ``fetch_k`` and are fused with reciprocal
        rank fusion: a document's score is the weighted sum of ``1 / (rrf_k + rank)``
        across the two lists. RRF uses rank, not raw scores, so the two very
        different scales never have to be reconciled. ``vector_weight`` in [0, 1]
        tilts the blend (1.0 = pure vector, 0.0 = pure keyword).
        """
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be between 0 and 1")
        vector = self._embedding.embed_query(query)
        clusters = self._probe(vector, kwargs.get("nprobe"))
        v_rows = self._store.search(vector, fetch_k, self.metric, filter=filter, clusters=clusters)
        k_rows = self._store.keyword_search(query, fetch_k, filter=filter)

        scores: dict = {}
        rows: dict = {}
        for rank, r in enumerate(v_rows):
            scores[r.id] = scores.get(r.id, 0.0) + vector_weight / (rrf_k + rank + 1)
            rows[r.id] = r
        for rank, r in enumerate(k_rows):
            scores[r.id] = scores.get(r.id, 0.0) + (1.0 - vector_weight) / (rrf_k + rank + 1)
            rows.setdefault(r.id, r)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            (Document(id=i, page_content=rows[i].content, metadata=dict(rows[i].metadata)), s)
            for i, s in ranked
        ]

    def ensure_fulltext_index(self) -> None:
        """Add the FULLTEXT index hybrid_search needs to an existing table.

        Tables created by 0.6.0+ already have it. Call this once after upgrading a
        table created by an earlier version, or when pointing at your own table.
        """
        self._store.ensure_fulltext_index()

    def get_by_ids(self, ids: Iterable[str]) -> List[Document]:
        ids = list(ids)
        found = {r.id: r for r in self._store.get(ids)}
        return [
            Document(id=i, page_content=found[i].content, metadata=dict(found[i].metadata))
            for i in ids
            if i in found
        ]

    def build_index(self, n_lists: int = 100, sample: int = 50000, iters: int = 10,
                    nprobe: int = 8, seed: int = 0) -> None:
        """Build an approximate IVF index so searches scan a fraction of the rows.

        Runs k-means over the stored vectors (sampled if there are more than
        ``sample``), assigns each row to its nearest centroid, and persists both.
        After this, searches only scan the ``nprobe`` centroids nearest the query.
        Recall is < 100% and rises with ``nprobe``. Call again to rebuild.
        """
        pairs = self._store.all_embeddings()
        if not pairs:
            raise ValueError("nothing to index yet; add documents first")
        vectors = [v for _, v in pairs]
        if len(vectors) > sample:
            vectors = random.Random(seed).sample(vectors, sample)
        n_lists = min(n_lists, len(vectors))
        centroids = _ivf.kmeans(vectors, n_lists, iters=iters, seed=seed)
        assignments = {rid: _ivf.nearest(v, centroids) for rid, v in pairs}
        self._store.write_index(centroids, assignments)
        self._centroids = centroids
        self._centroids_loaded = True
        self._nprobe = nprobe

    def _current_centroids(self) -> Optional[List[List[float]]]:
        if not self._centroids_loaded:
            self._centroids = self._store.read_centroids()
            self._centroids_loaded = True
        return self._centroids

    def _probe(self, embedding: List[float], nprobe: Optional[int]) -> Optional[List[int]]:
        centroids = self._current_centroids()
        if not centroids:
            return None
        return _ivf.nearest_n(embedding, centroids, nprobe or self._nprobe)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        clusters = self._probe(embedding, kwargs.get("nprobe"))
        rows = self._store.search(embedding, fetch_k, self.metric, filter=filter,
                                  with_vector=True, clusters=clusters)
        candidates: List[List[float]] = []
        kept = []
        for r in rows:
            if r.embedding is not None:
                candidates.append(r.embedding)
                kept.append(r)
        if not candidates:
            return []
        picks = _mmr(embedding, candidates, k, lambda_mult)
        return [
            Document(id=kept[i].id, page_content=kept[i].content, metadata=dict(kept[i].metadata))
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

    @staticmethod
    def _euclidean_relevance_score_fn(score: float) -> float:
        # score is (1 - euclidean_distance) from the wrapper, so distance = 1 - score
        # and is >= 0. Map to (0, 1] with 1 / (1 + distance): closer -> higher, bounded.
        distance = 1.0 - score
        return 1.0 / (1.0 + max(0.0, distance))

    def _select_relevance_score_fn(self):
        if self.metric == "cosine":
            return self._cosine_relevance_score_fn
        if self.metric == "euclidean":
            return self._euclidean_relevance_score_fn
        # dot: DISTANCE(...,'DOT') isn't a bounded distance and the docs don't pin
        # down its sign, so there's no honest [0, 1] mapping yet. super() raises.
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
        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)
    return selected
