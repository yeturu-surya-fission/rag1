import os
from ragpipeline import setup, generate_answer

def run_tests():
    print("--- 1. Setting up RAG pipeline ---")
    index, chunks, bm25 = setup()
    print(f"Total chunks in corpus: {len(chunks)}")

    test_queries = [
        "What is the maternity leave policy?",
        "How many casual leaves are allowed in a year?",
        "What are the working hours and attendance policies?",
    ]

    print("\n--- 2. Testing Live Answer Generation (No Cache) ---")
    for q in test_queries:
        print(f"\nQuery: {q}")
        ans = generate_answer(q, index, chunks, bm25)
        print("Answer:\n", ans)
        print("=" * 60)

    # Verify no cache files are created in storage
    print("\n--- 3. Verifying Cache Absence ---")
    query_cache_exists = os.path.exists("storage/query_cache.json")
    semantic_cache_exists = os.path.exists("storage/semantic_cache.json")
    print(f"query_cache.json exists: {query_cache_exists}")
    print(f"semantic_cache.json exists: {semantic_cache_exists}")
    assert not query_cache_exists, "query_cache.json should not exist!"
    assert not semantic_cache_exists, "semantic_cache.json should not exist!"
    print("Verification complete: Caching is completely removed and generation is working!")

if __name__ == "__main__":
    run_tests()