import json
import os
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

CACHE_PATH = "storage/semantic_cache.json"
SIMILARITY_THRESHOLD = 0.85
INTENT_TERMS = {
    "relationship": {"love", "loves", "like", "likes", "marry", "married", "friend", "friends"},
    "identity": {"who", "what", "is"},
    "availability": {"available", "many", "much", "days", "hours"},
}
IGNORED_TERMS = {
    "a", "an", "and", "are", "does", "for", "he", "how", "is", "it",
    "many", "of", "the", "to", "was", "what", "who", "with",
}


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return []

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    return cache if isinstance(cache, list) else []


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_embedding(text, embed_model):
    return embed_model.embed_query(text)


def add_to_cache(query, embedding, answer):
    """Create a JSON-safe semantic-cache entry."""
    embedding_array = np.asarray(embedding, dtype=float).reshape(-1)
    return {
        "query": query,
        "embedding": embedding_array.tolist(),
        "answer": answer,
    }


def meaningful_terms(query):
    terms = re.findall(r"[a-z0-9]+", query.lower())
    return {term for term in terms if term not in IGNORED_TERMS and len(term) > 2}


def query_intent(query):
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if terms.intersection(INTENT_TERMS["relationship"]):
        return "relationship"
    if "who" in terms and ("does" in terms or "do" in terms):
        return "relationship"
    if terms.intersection(INTENT_TERMS["availability"]):
        return "availability"
    if "who" in terms or "what" in terms:
        return "identity"
    return "general"


def find_similar(query, query_embedding, cache):
    """Return a cached answer only when semantic and lexical checks agree."""
    if not cache:
        return None

    query_vector = np.asarray(query_embedding, dtype=float).reshape(1, -1)
    query_terms = meaningful_terms(query)
    best_score = 0.0
    best_answer = None

    for item in cache:
        if not isinstance(item, dict) or not item.get("answer"):
            continue

        if query_intent(query) != query_intent(item.get("query", "")):
            continue

        stored_vector = np.asarray(item["embedding"], dtype=float).reshape(1, -1)
        if stored_vector.shape[1] != query_vector.shape[1]:
            continue

        cached_terms = meaningful_terms(item.get("query", ""))
        if query_terms and cached_terms and not query_terms.intersection(cached_terms):
            continue

        score = float(cosine_similarity(query_vector, stored_vector)[0, 0])

        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_score >= SIMILARITY_THRESHOLD:
        print(f"Semantic Cache hit ⚡ (score: {best_score:.2f})")
        return best_answer

    print(f"Semantic Cache miss ❌ (best score: {best_score:.2f})")
    return None