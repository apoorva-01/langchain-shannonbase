"""SQL for MySQL 9's native VECTOR type (ShannonBase / MySQL / HeatWave).

Kept as pure string builders so they can be unit-tested without a database.
Embeddings are passed to MySQL as a JSON array string via STRING_TO_VECTOR(),
and cosine distance comes from the native DISTANCE(a, b, 'COSINE') function.
similarity = 1 - distance.
"""

from __future__ import annotations

import json
import re
from typing import List

# Distance metric -> MySQL DISTANCE() metric name.
_METRICS = {"cosine": "COSINE", "dot": "DOT", "euclidean": "EUCLIDEAN"}

# Metadata keys go into the SQL text (as a JSON path), so they must be identifiers.
# Values are always bound as parameters, never interpolated.
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def vector_literal(embedding: List[float]) -> str:
    """Serialize a vector to the JSON-array string STRING_TO_VECTOR expects."""
    return json.dumps([float(x) for x in embedding])


def create_table_sql(table: str, dim: int) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS `{table}` ("
        "  id VARCHAR(36) PRIMARY KEY,"
        "  content TEXT,"
        "  metadata JSON,"
        f"  embedding VECTOR({dim})"
        ")"
    )


def insert_sql(table: str) -> str:
    """Upsert one row. Embedding bind param is wrapped in STRING_TO_VECTOR()."""
    return (
        f"INSERT INTO `{table}` (id, content, metadata, embedding) "
        "VALUES (%s, %s, %s, STRING_TO_VECTOR(%s)) "
        "AS new ON DUPLICATE KEY UPDATE "
        "content = new.content, metadata = new.metadata, embedding = new.embedding"
    )


def search_sql(table: str, metric: str = "cosine", filter_keys=(),
               with_vector: bool = False, n_clusters: int = 0) -> str:
    m = _METRICS.get(metric)
    if m is None:
        raise ValueError(f"unknown metric {metric!r}; use one of {list(_METRICS)}")
    for key in filter_keys:
        if not _KEY.match(key):
            raise ValueError(f"invalid filter key {key!r}; must be an identifier")
    cols = "id, content, metadata"
    if with_vector:
        cols += ", VECTOR_TO_STRING(embedding) AS emb"
    parts = [f"metadata->>'$.{key}' = %s" for key in filter_keys]
    if n_clusters:
        parts.append("cluster IN (" + ", ".join(["%s"] * n_clusters) + ")")
    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    return (
        f"SELECT {cols}, "
        f"DISTANCE(embedding, STRING_TO_VECTOR(%s), '{m}') AS dist "
        f"FROM `{table}`{where} ORDER BY dist ASC LIMIT %s"
    )


def all_embeddings_sql(table: str) -> str:
    return f"SELECT id, VECTOR_TO_STRING(embedding) FROM `{table}`"


def create_ivf_table_sql(table: str, dim: int) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS `{table}_ivf` ("
        "  cid INT PRIMARY KEY,"
        f"  centroid VECTOR({dim})"
        ")"
    )


def insert_ivf_sql(table: str) -> str:
    return f"INSERT INTO `{table}_ivf` (cid, centroid) VALUES (%s, STRING_TO_VECTOR(%s))"


def clear_ivf_sql(table: str) -> str:
    return f"DELETE FROM `{table}_ivf`"


def select_centroids_sql(table: str) -> str:
    return f"SELECT cid, VECTOR_TO_STRING(centroid) FROM `{table}_ivf` ORDER BY cid ASC"


def add_cluster_column_sql(table: str) -> str:
    return f"ALTER TABLE `{table}` ADD COLUMN cluster INT"


def add_cluster_index_sql(table: str) -> str:
    return f"ALTER TABLE `{table}` ADD INDEX idx_cluster (cluster)"


def update_cluster_sql(table: str) -> str:
    return f"UPDATE `{table}` SET cluster = %s WHERE id = %s"


def delete_sql(table: str, n_ids: int) -> str:
    placeholders = ", ".join(["%s"] * n_ids)
    return f"DELETE FROM `{table}` WHERE id IN ({placeholders})"


def select_by_ids_sql(table: str, n_ids: int) -> str:
    placeholders = ", ".join(["%s"] * n_ids)
    return f"SELECT id, content, metadata FROM `{table}` WHERE id IN ({placeholders})"


def distance_to_score(distance: float) -> float:
    """Convert a cosine distance to a [0, 1]-ish similarity score."""
    return 1.0 - float(distance)
