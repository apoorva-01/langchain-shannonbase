"""End-to-end RAG on MySQL 9 / ShannonBase in about 20 lines.

    pip install "langchain-shannonbase[mysql]" langchain-openai
    export OPENAI_API_KEY=...            # for embeddings + the LLM
    export SB_HOST=127.0.0.1 SB_USER=root SB_PASSWORD= SB_DATABASE=rag

Run a ShannonBase container to get MySQL 9 vector features locally:
https://github.com/Shannon-Data/ShannonBase
"""
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_shannonbase import ShannonBaseVectorStore

store = ShannonBaseVectorStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    table="kb",
    host=os.getenv("SB_HOST", "127.0.0.1"),
    port=int(os.getenv("SB_PORT", "3306")),
    user=os.getenv("SB_USER", "root"),
    password=os.getenv("SB_PASSWORD", ""),
    database=os.getenv("SB_DATABASE", "rag"),
)

store.add_texts(
    [
        "Refunds are accepted within 30 days of purchase.",
        "Shipping is free on orders over $50.",
        "Support is available Monday to Friday, 9am to 5pm.",
    ]
)

question = "How long do I have to return something?"
context = "\n".join(doc.page_content for doc in store.similarity_search(question, k=2))

llm = ChatOpenAI(model="gpt-4o-mini")
answer = llm.invoke(f"Answer using only this context:\n{context}\n\nQuestion: {question}")
print(answer.content)
