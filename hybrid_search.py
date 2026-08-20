from vector_store import search as faiss_search
from bm25_search import bm25_search


def hybrid_search(query, index, bm25, chunks, embed_model, top_k=5, query_vector=None):
    """
    Combine FAISS + BM25 results
    """

    # ---- FAISS ----
    if query_vector is None:
        query_vector = embed_model.embed_query(query)
    faiss_results = faiss_search(index, query_vector, chunks, top_k=top_k)

    # ---- BM25 ----
    bm25_results = bm25_search(query, bm25, chunks, top_k=top_k)

    # ---- Combine ----
    combined = []

    seen = set()

    for doc in faiss_results + bm25_results:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)

    # return top_k combined
    return combined[:top_k]