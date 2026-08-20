import os
from pathlib import Path

from dotenv import load_dotenv
from chunking import load_and_chunk_pdf
from embedings import create_embeddings
from vector_store import build_faiss_index, search

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from hybrid_search import hybrid_search
from bm25_search import build_bm25
from storage import (
    save_index,
    load_index,
    save_chunks,
    load_chunks,
    index_exists,
    save_manifest,
)

import faiss
import numpy as np
from cache import (
    load_cache as load_query_cache,
    save_cache as save_query_cache,
    normalize_query,
)
from semantic_cache import (
    load_cache as load_semantic_cache,
    save_cache as save_semantic_cache,
    add_to_cache,
    find_similar,
)

cache = load_query_cache()
semantic_cache = load_semantic_cache()
load_dotenv()


def setup():
    pdf_paths = sorted(Path("data").glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in the data directory.")

    manifest = {
        str(path): {
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in pdf_paths
    }

    if index_exists(manifest):
        print("Loading existing index...")
        index = load_index()
        chunks = load_chunks()
    else:
        print(f"Creating index for {len(pdf_paths)} PDF file(s)...")
        chunks = []
        for pdf_path in pdf_paths:
            print(f"Chunking {pdf_path.name}...")
            chunks.extend(load_and_chunk_pdf(str(pdf_path)))

        vectors = create_embeddings(chunks)
        index = build_faiss_index(chunks, vectors)
        save_index(index)
        save_chunks(chunks)
        save_manifest(manifest)

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
    api_key = get_gemini_api_key()
    top_k = get_top_k()

    embed_model = GoogleGenerativeAIEmbeddings(
        model=get_setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview"),
        api_key=api_key,
    )
    normalized_query = normalize_query(query)

    # Check the exact cache before making any API request.
    if normalized_query in cache:
        print("Cache hit ⚡")
        return cache[normalized_query]

    # One embedding powers both semantic-cache lookup and FAISS retrieval.
    query_embedding = embed_model.embed_query(query)
    cached_answer = find_similar(query, query_embedding, semantic_cache)
    if cached_answer:
        return cached_answer

    print("Cache miss -> running RAG")

    # One retrieval pass: FAISS uses the existing vector and BM25 uses the text.
    final_chunks = hybrid_search(
        query,
        index,
        bm25,
        chunks,
        embed_model,
        top_k=top_k,
        query_vector=query_embedding,
    )

    # ---- Step 6: Build Context ----
    context = "\n\n".join([c.page_content for c in final_chunks])

    # ---- Step 7: LLM ----
    llm = ChatGoogleGenerativeAI(
        model=get_setting("GEMINI_GENERATION_MODEL", "gemini-3.1-flash-lite"),
        api_key=api_key,
    )

    prompt = f"""
You are an assistant answering questions from company policy documents.

Rules:
1. Answer ONLY what the user asked
2. Be concise and direct
3. Do NOT list all related information unless asked
4. If the question is specific, give a specific answer
5. Use bullet points ONLY if needed
6. If multiple types exist, summarize briefly

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = extract_text_content(response.content).strip()
    cache[normalized_query] = answer
    save_query_cache(cache)
    semantic_cache.append(add_to_cache(query, query_embedding, answer))
    save_semantic_cache(semantic_cache)
    print(f"Cache size: {len(cache)}")
    


    return answer
    
    

    

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