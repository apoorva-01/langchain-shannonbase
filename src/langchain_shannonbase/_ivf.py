"""A small IVF (inverted file) index, in pure Python.

k-means gives coarse centroids; every stored vector is assigned to its nearest
centroid. A search only scans the rows in the `nprobe` centroids closest to the
query, instead of the whole table. This is approximate: recall rises with nprobe.
"""
from __future__ import annotations

import random
from typing import List


def _dist2(a: List[float], b: List[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def nearest(vec: List[float], centroids: List[List[float]]) -> int:
    best, best_d = 0, None
    for i, c in enumerate(centroids):
        d = _dist2(vec, c)
        if best_d is None or d < best_d:
            best_d, best = d, i
    return best


def nearest_n(vec: List[float], centroids: List[List[float]], n: int) -> List[int]:
    order = sorted(range(len(centroids)), key=lambda i: _dist2(vec, centroids[i]))
    return order[: max(1, n)]


def kmeans(vectors: List[List[float]], k: int, iters: int = 10, seed: int = 0) -> List[List[float]]:
    """Return up to k centroids. Deterministic for a given seed."""
    rng = random.Random(seed)
    if k >= len(vectors):
        return [list(v) for v in vectors]
    centroids = [list(v) for v in rng.sample(vectors, k)]
    for _ in range(iters):
        buckets: List[List[List[float]]] = [[] for _ in range(k)]
        for v in vectors:
            buckets[nearest(v, centroids)].append(v)
        for i, bucket in enumerate(buckets):
            if bucket:
                centroids[i] = [sum(col) / len(bucket) for col in zip(*bucket)]
    return centroids
