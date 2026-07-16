"""SQL for MySQL 9's native VECTOR type (ShannonBase / MySQL / HeatWave).

Pure string builders so they can be unit-tested without a database. Column names
come from a Schema (validated as identifiers); values are always bound, never
interpolated. Embeddings go in via STRING_TO_VECTOR() and cosine distance comes
from the native DISTANCE(a, b, 'COSINE') function; similarity = 1 - distance.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

_METRICS = {"cosine": "COSINE", "dot": "DOT", "euclidean": "EUCLIDEAN"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"invalid SQL identifier {name!r}; must match [A-Za-z_][A-Za-z0-9_]*")
    return name


@dataclass
class Schema:
    """Table and column names. Defaults match the schema the store creates."""

    table: str
    id: str = "id"
    content: str = "content"
    metadata: str = "metadata"
    embedding: str = "embedding"
    cluster: str = "cluster"

    def __post_init__(self):
        for name in (self.table, self.id, self.content, self.metadata, self.embedding, self.cluster):
            ident(name)

    @property
    def ivf_table(self) -> str:
        return f"{self.table}_ivf"


def vector_literal(embedding: List[float]) -> str:
    return json.dumps([float(x) for x in embedding])


def create_table_sql(s: Schema, dim: int) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS `{s.table}` ("
        f"  `{s.id}` VARCHAR(36) PRIMARY KEY,"
        f"  `{s.content}` TEXT,"
        f"  `{s.metadata}` JSON,"
        f"  `{s.embedding}` VECTOR({dim})"
        ")"
    )


def insert_sql(s: Schema) -> str:
    return (
        f"INSERT INTO `{s.table}` (`{s.id}`, `{s.content}`, `{s.metadata}`, `{s.embedding}`) "
        "VALUES (%s, %s, %s, STRING_TO_VECTOR(%s)) "
        "AS new ON DUPLICATE KEY UPDATE "
        f"`{s.content}` = new.`{s.content}`, `{s.metadata}` = new.`{s.metadata}`, "
        f"`{s.embedding}` = new.`{s.embedding}`"
    )


def search_sql(s: Schema, metric: str = "cosine", filter_clauses=(),
               with_vector: bool = False, n_clusters: int = 0) -> str:
    m = _METRICS.get(metric)
    if m is None:
        raise ValueError(f"unknown metric {metric!r}; use one of {list(_METRICS)}")
    cols = f"`{s.id}`, `{s.content}`, `{s.metadata}`"
    if with_vector:
        cols += f", VECTOR_TO_STRING(`{s.embedding}`) AS emb"
    parts = list(filter_clauses)
    if n_clusters:
        parts.append(f"`{s.cluster}` IN (" + ", ".join(["%s"] * n_clusters) + ")")
    where = (" WHERE " + " AND ".join(parts)) if parts else ""
    return (
        f"SELECT {cols}, "
        f"DISTANCE(`{s.embedding}`, STRING_TO_VECTOR(%s), '{m}') AS dist "
        f"FROM `{s.table}`{where} ORDER BY dist ASC LIMIT %s"
    )


def delete_sql(s: Schema, n_ids: int) -> str:
    placeholders = ", ".join(["%s"] * n_ids)
    return f"DELETE FROM `{s.table}` WHERE `{s.id}` IN ({placeholders})"


def select_by_ids_sql(s: Schema, n_ids: int) -> str:
    placeholders = ", ".join(["%s"] * n_ids)
    return f"SELECT `{s.id}`, `{s.content}`, `{s.metadata}` FROM `{s.table}` WHERE `{s.id}` IN ({placeholders})"


def all_embeddings_sql(s: Schema) -> str:
    return f"SELECT `{s.id}`, VECTOR_TO_STRING(`{s.embedding}`) FROM `{s.table}`"


def create_ivf_table_sql(s: Schema, dim: int) -> str:
    return f"CREATE TABLE IF NOT EXISTS `{s.ivf_table}` (cid INT PRIMARY KEY, centroid VECTOR({dim}))"


def insert_ivf_sql(s: Schema) -> str:
    return f"INSERT INTO `{s.ivf_table}` (cid, centroid) VALUES (%s, STRING_TO_VECTOR(%s))"


def clear_ivf_sql(s: Schema) -> str:
    return f"DELETE FROM `{s.ivf_table}`"


def select_centroids_sql(s: Schema) -> str:
    return f"SELECT cid, VECTOR_TO_STRING(centroid) FROM `{s.ivf_table}` ORDER BY cid ASC"


def add_cluster_column_sql(s: Schema) -> str:
    return f"ALTER TABLE `{s.table}` ADD COLUMN `{s.cluster}` INT"


def add_cluster_index_sql(s: Schema) -> str:
    return f"ALTER TABLE `{s.table}` ADD INDEX `idx_{s.cluster}` (`{s.cluster}`)"


def update_cluster_sql(s: Schema) -> str:
    return f"UPDATE `{s.table}` SET `{s.cluster}` = %s WHERE `{s.id}` = %s"


def distance_to_score(distance: float) -> float:
    return 1.0 - float(distance)
