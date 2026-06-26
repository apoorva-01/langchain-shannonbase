"""SQL for MySQL 9's native VECTOR type (ShannonBase / MySQL / HeatWave).

Kept as pure string builders so they can be unit-tested without a database.
Embeddings are passed to MySQL as a JSON array string via STRING_TO_VECTOR(),
and cosine distance comes from the native DISTANCE(a, b, 'COSINE') function.
similarity = 1 - distance.
"""

from __future__ import annotations

import json
from typing import List

# Distance metric -> MySQL DISTANCE() metric name.
_METRICS = {"cosine": "COSINE", "dot": "DOT", "euclidean": "EUCLIDEAN"}


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


def search_sql(table: str, metric: str = "cosine") -> str:
    m = _METRICS.get(metric)
    if m is None:
        raise ValueError(f"unknown metric {metric!r}; use one of {list(_METRICS)}")
    return (
        "SELECT id, content, metadata, "
        f"DISTANCE(embedding, STRING_TO_VECTOR(%s), '{m}') AS dist "
        f"FROM `{table}` ORDER BY dist ASC LIMIT %s"
    )


def delete_sql(table: str, n_ids: int) -> str:
    placeholders = ", ".join(["%s"] * n_ids)
    return f"DELETE FROM `{table}` WHERE id IN ({placeholders})"


def distance_to_score(distance: float) -> float:
    """Convert a cosine distance to a [0, 1]-ish similarity score."""
    return 1.0 - float(distance)
