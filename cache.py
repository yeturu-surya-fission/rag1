import json
import os

CACHE_PATH = "storage/query_cache.json"


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def normalize_query(query: str) -> str:
    return query.lower().strip()