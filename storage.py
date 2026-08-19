import os
import json
import faiss
from langchain_core.documents import Document


INDEX_PATH = "storage/faiss_index.bin"
CHUNKS_PATH = "storage/chunks.json"


def ensure_storage_dir():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)


def save_index(index):
    ensure_storage_dir()
    faiss.write_index(index, INDEX_PATH)


def load_index():
    return faiss.read_index(INDEX_PATH)


def save_chunks(chunks):
    ensure_storage_dir()
    data = [c.page_content for c in chunks]

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [Document(page_content=t) for t in data]


def index_exists():
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)