from rank_bm25 import BM25Okapi


import re

STOPWORDS = {
    "the", "is", "are", "in", "on", "how", "many", "a", "an", "of", "for"
}

def preprocess(text):
    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    tokens = text.split()

    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens


def build_bm25(chunks):
    """
    Create BM25 index from chunks
    """

    tokenized_corpus = [
        preprocess(chunk.page_content) for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, tokenized_corpus


def bm25_search(query, bm25, chunks, top_k=5):
    """
    Search using BM25
    """

    tokenized_query = preprocess(query)

    scores = bm25.get_scores(tokenized_query)

    # Get top indices
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = [chunks[i] for i in top_indices]

    return results