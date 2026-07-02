"""Minimal RAG-style usage against a real ShannonBase / MySQL 9 / HeatWave DB.

    pip install 'langchain-shannonbase[mysql]' langchain-openai
    export OPENAI_API_KEY=...
    python examples/basic.py
"""

from langchain_openai import OpenAIEmbeddings

from langchain_shannonbase import ShannonBaseVectorStore

store = ShannonBaseVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    table="documents",
    host="127.0.0.1",
    port=3306,
    user="root",
    password="",
    database="rag",
)

store.add_texts(
    [
        "Our refund policy allows returns within 30 days of purchase.",
        "Support is available Monday to Friday, 9am to 6pm.",
        "Free shipping on orders over $50.",
    ],
    metadatas=[{"topic": "refunds"}, {"topic": "support"}, {"topic": "shipping"}],
)

# Use it like any LangChain vector store:
retriever = store.as_retriever(search_kwargs={"k": 2})
for doc in retriever.invoke("how long do I have to return something?"):
    print(f"- {doc.page_content}  ({doc.metadata})")
