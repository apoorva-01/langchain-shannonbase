# langchain-shannonbase

A [LangChain](https://python.langchain.com) `VectorStore` for **MySQL 9's native
`VECTOR` type** — so you can do RAG on a database you already run.

Works with **[ShannonBase](https://github.com/Shannon-Data/ShannonBase)** (the
open-source MySQL-for-AI), **self-hosted MySQL 9**, and **MySQL HeatWave** — they
all share the same `VECTOR` / `STRING_TO_VECTOR` / `DISTANCE` surface.

## Why this exists

If your data already lives in MySQL, your options for LangChain vector search were
thin: the only MySQL `VectorStore` is locked to Google Cloud SQL, and ShannonBase's
LangChain integration was on its wishlist but unbuilt. This fills that gap — a
plain, self-hostable adapter that plugs MySQL 9 into the LangChain ecosystem, no
separate vector database required.

## Install

```bash
pip install "langchain-shannonbase[mysql]"
```

