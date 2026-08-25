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

load_dotenv(override=True)


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

    print(f"Retrieving context for query: '{query}'...")

    # Embed query and perform hybrid search
    query_embedding = embed_model.embed_query(query)

    final_chunks = hybrid_search(
        query,
        index,
        bm25,
        chunks,
        embed_model,
        top_k=top_k,
        query_vector=query_embedding,
    )

    # Build Context from retrieved chunks
    context = "\n\n".join([c.page_content for c in final_chunks])

    # LLM generation
    llm = ChatGoogleGenerativeAI(
        model=get_setting("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
        api_key=api_key,
    )

    prompt = f"""
You are an assistant answering questions from company policy and documentation.

Rules:
1. Answer accurately based on the provided Context.
2. If the answer cannot be found in the context, clearly state that the information is not available in the provided documents.
3. Be concise, direct, and well-structured.
4. If multiple points or types exist, summarize clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    print("Generating response...")
    response = llm.invoke(prompt)
    answer = extract_text_content(response.content).strip()

    return answer


if __name__ == "__main__":
    index, chunks, bm25 = setup()
    print("RAG Ready (No Cache Mode)\n")

    while True:
        try:
            q = input("Ask: ")
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() == "exit":
            break

        ans = generate_answer(q, index, chunks, bm25)
        print("\nAnswer:", ans)
        print("-" * 50)