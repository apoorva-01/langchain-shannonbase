"""Live integration test against a real MySQL 9 / ShannonBase / HeatWave instance.

Skipped unless SHANNONBASE_TEST_DSN-style env vars are set, so CI stays green
without provisioning a database. To run locally against a ShannonBase or MySQL 9:

    export SB_HOST=127.0.0.1 SB_PORT=3306 SB_USER=root SB_PASSWORD=... SB_DATABASE=test
    pytest tests/test_integration.py -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SB_HOST"),
    reason="set SB_HOST/SB_USER/SB_PASSWORD/SB_DATABASE to run live DB tests",
)


def _conn_kwargs():
    return dict(
        host=os.environ["SB_HOST"],
        port=int(os.environ.get("SB_PORT", 3306)),
        user=os.environ["SB_USER"],
        password=os.environ.get("SB_PASSWORD", ""),
        database=os.environ["SB_DATABASE"],
    )


def test_roundtrip_against_real_db():
    import hashlib
    import re

    from langchain_core.embeddings import Embeddings

    from langchain_shannonbase import ShannonBaseVectorStore

    tok = re.compile(r"[a-z0-9]+")

    class E(Embeddings):
        def _e(self, t):
            v = [0.0] * 64
            for w in tok.findall(t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            return v

        def embed_documents(self, x):
            return [self._e(t) for t in x]

        def embed_query(self, t):
            return self._e(t)

    vs = ShannonBaseVectorStore(embedding=E(), table="lc_sb_itest", **_conn_kwargs())
    vs.add_texts(["reset my password", "cancel my subscription"], ids=["a", "b"])
    docs = vs.similarity_search("how do I reset my password", k=1)
    assert docs[0].page_content == "reset my password"
    vs.delete(["a", "b"])
