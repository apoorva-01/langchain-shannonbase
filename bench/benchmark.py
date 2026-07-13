"""Latency benchmark for ShannonBaseVectorStore.

Search is exact (a full DISTANCE scan), so recall is 1.0 by construction. What
this measures is how query latency scales with the number of stored vectors.

    # in-memory reference (no database, pure-Python scan):
    python bench/benchmark.py --n 10000 --backend memory

    # real numbers against your instance:
    export SB_HOST=127.0.0.1 SB_USER=root SB_PASSWORD= SB_DATABASE=bench
    python bench/benchmark.py --n 10000 --backend mysql

Report the numbers from a real instance. The in-memory backend reflects Python
scan cost, not MySQL, so treat it as a shape, not a headline.
"""
import argparse
import os
import random
import statistics
import time

from langchain_core.embeddings import Embeddings

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore


class RandomEmbeddings(Embeddings):
    """Deterministic pseudo-random vectors keyed off the text, for reproducibility."""

    def __init__(self, dim):
        self.dim = dim

    def _vec(self, text):
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--backend", choices=["memory", "mysql"], default="memory")
    args = ap.parse_args()

    emb = RandomEmbeddings(args.dim)
    if args.backend == "memory":
        store = ShannonBaseVectorStore(embedding=emb, store=InMemoryStore())
        label = "in-memory (Python reference)"
    else:
        store = ShannonBaseVectorStore(
            embedding=emb, table="bench",
            host=os.getenv("SB_HOST", "127.0.0.1"), port=int(os.getenv("SB_PORT", "3306")),
            user=os.getenv("SB_USER", "root"), password=os.getenv("SB_PASSWORD", ""),
            database=os.getenv("SB_DATABASE", "bench"),
        )
        label = "MySQL / ShannonBase"

    texts = [f"doc-{i}" for i in range(args.n)]
    t0 = time.perf_counter()
    for start in range(0, args.n, 500):
        store.add_texts(texts[start:start + 500])
    insert_s = time.perf_counter() - t0

    latencies = []
    for i in range(args.queries):
        q = f"query-{i}"
        t = time.perf_counter()
        store.similarity_search(q, k=args.k)
        latencies.append((time.perf_counter() - t) * 1000)

    print(f"backend:   {label}")
    print(f"vectors:   {args.n:,} x {args.dim}d   k={args.k}   queries={args.queries}")
    print(f"insert:    {insert_s:.1f}s  ({args.n / insert_s:,.0f} vectors/s)")
    print(f"query p50: {percentile(latencies, 50):.1f} ms")
    print(f"query p95: {percentile(latencies, 95):.1f} ms")
    print(f"query avg: {statistics.mean(latencies):.1f} ms")


if __name__ == "__main__":
    main()
