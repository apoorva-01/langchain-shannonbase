"""LangChain's standard VectorStore integration suite, run offline.

The suite exercises add/get_by_ids/delete/search against a real store. We point it
at InMemoryStore so it runs in CI without provisioning a database — same behavior
as the MySQL backend, just in-process.
"""

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_shannonbase import InMemoryStore, ShannonBaseVectorStore


class TestShannonBaseVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> VectorStore:
        return ShannonBaseVectorStore(embedding=self.get_embeddings(), store=InMemoryStore())
