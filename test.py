from chunking import load_and_chunk_pdf
from bm25_search import build_bm25, bm25_search

chunks = load_and_chunk_pdf("data/company-policy.pdf")

bm25, _ = build_bm25(chunks)

query = "how many leaves are allowed in total"

results = bm25_search(query, bm25, chunks, top_k=5)

for r in results:
    print(r.page_content)
    print("-" * 50)