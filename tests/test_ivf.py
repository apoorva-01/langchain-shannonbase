"""IVF index tests, offline: correctness (probing every list == exact) and a
measured recall@k for a small nprobe. Vectors are unit-normalized so the k-means
(Euclidean) clustering and the cosine search agree.
"""
import math
import random

import pytest
from langchain_core.embeddings import Embeddings

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore

D, C, PER = 16, 8, 40
rng = random.Random(0)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


centers = [_unit([rng.uniform(-1, 1) for _ in range(D)]) for _ in range(C)]
_texts, _lookup = [], {}
for ci, c in enumerate(centers):
    for j in range(PER):
        t = f"pt-{ci}-{j}"
        _texts.append(t)
        _lookup[t] = _unit([x + rng.gauss(0, 0.15) for x in c])

_queries = []
for i in range(30):
    qt = f"q-{i}"
    _queries.append(qt)
    _lookup[qt] = _unit([x + rng.gauss(0, 0.15) for x in centers[i % C]])


class LookupEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [_lookup[t] for t in texts]

    def embed_query(self, text):
        return _lookup[text]


def _fresh():
    vs = ShannonBaseVectorStore(embedding=LookupEmbeddings(), store=InMemoryStore())
    vs.add_texts(_texts, ids=_texts)
    return vs


def _recall(vs, exact, k, nprobe):
    hit = tot = 0
    for qt in _queries:
        approx = {d.id for d in vs.similarity_search(qt, k=k, nprobe=nprobe)}
        hit += len(exact[qt] & approx)
        tot += len(exact[qt])
    return hit / tot


def test_probing_all_lists_matches_exact():
    vs = _fresh()
    exact = {qt: {d.id for d in vs.similarity_search(qt, k=10)} for qt in _queries}
    vs.build_index(n_lists=C, nprobe=C, iters=15, seed=0)
    assert _recall(vs, exact, k=10, nprobe=C) == 1.0


def test_small_nprobe_keeps_high_recall():
    vs = _fresh()
    exact = {qt: {d.id for d in vs.similarity_search(qt, k=10)} for qt in _queries}
    vs.build_index(n_lists=C, nprobe=3, iters=15, seed=0)
    assert _recall(vs, exact, k=10, nprobe=3) >= 0.75


def test_build_index_on_empty_raises():
    vs = ShannonBaseVectorStore(embedding=LookupEmbeddings(), store=InMemoryStore())
    with pytest.raises(ValueError):
        vs.build_index()
