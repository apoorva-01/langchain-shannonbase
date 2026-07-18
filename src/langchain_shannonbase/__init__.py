"""langchain-shannonbase — a LangChain VectorStore for MySQL 9's VECTOR type."""

__version__ = "0.8.0"

from ._store import InMemoryStore, MySQLStore
from .vectorstores import ShannonBaseVectorStore


def __getattr__(name):
    # AsyncMySQLStore lives behind the optional [async] extra; import it lazily so
    # `import langchain_shannonbase` doesn't require aiomysql to be installed.
    if name == "AsyncMySQLStore":
        from ._async_store import AsyncMySQLStore
        return AsyncMySQLStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ShannonBaseVectorStore", "MySQLStore", "InMemoryStore",
           "AsyncMySQLStore", "__version__"]
