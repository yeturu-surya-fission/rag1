import os

from dotenv import load_dotenv
from chunking import load_and_chunk_pdf
from embedings import create_embeddings
from vector_store import build_faiss_index, search

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from hybrid_search import hybrid_search
from bm25_search import build_bm25
from query_rewriter import rewrite_query
from reranker import rerank_chunks
from storage import (
    save_index,
    load_index,
    save_chunks,
    load_chunks,
    index_exists,
)

import faiss
import numpy as np

load_dotenv()


def setup():
    file_path = "data/company-policy.pdf"

    if index_exists():
        print("Loading existing index...")
        index = load_index()
        chunks = load_chunks()
    else:
        print("Creating new index...")
        chunks = load_and_chunk_pdf(file_path)
        vectors = create_embeddings(chunks)
        index = build_faiss_index(chunks, vectors)
        save_index(index)
        save_chunks(chunks)

    bm25, _ = build_bm25(chunks)
    return index, chunks, bm25

def get_setting(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def get_gemini_api_key():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
    return api_key


def get_top_k(default=5):
    try:
        return int(get_setting("TOP_K", str(default)))
    except ValueError:
        return default


# -------- Step 1: Build / Load Index -------- #

# -------- Step 2: Generate Answer -------- #

def extract_text_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    texts.append(item["content"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                texts.append(item.text)
        if texts:
            return "\n".join(texts)

    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text

    return str(content)





def generate_answer(query, index, chunks, bm25):

    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # ---- Step 1: Rewrite ----
    rewritten = rewrite_query(query)

    print(f"\nOriginal: {query}")
    print(f"Rewritten: {rewritten}\n")

    # ---- Step 2: Retrieval ----
    res1 = hybrid_search(query, index, bm25, chunks, embed_model, top_k=5)
    res2 = hybrid_search(rewritten, index, bm25, chunks, embed_model, top_k=5)

    # ---- Step 3: Combine ----
    combined = []
    seen = set()

    for doc in res1 + res2:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)

    # ---- Step 4: Rerank ----
    reranked = rerank_chunks(query, combined)

    # ---- Step 5: Final Selection ----
    final_chunks = reranked[:5]

    # ---- Step 6: Build Context ----
    context = "\n\n".join([c.page_content for c in final_chunks])

    # ---- Step 7: LLM ----
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash"
    )

    prompt = f"""
You are an assistant that answers questions using the given context.

Rules:
1. Combine information from multiple parts if needed
2. Be accurate and concise
3. If no relevant info exists, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content

# -------- Step 3: Run Chat -------- #

if __name__ == "__main__":

    index, chunks, bm25 = setup()

    print("RAG Ready (Persistent Mode)\n")

    while True:
        q = input("Ask: ")

        if q.lower() == "exit":
            break

        ans = generate_answer(q, index, chunks, bm25)

        print("\nAnswer:", ans)
        print("-" * 50)