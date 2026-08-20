import os
import json
import faiss
from langchain_core.documents import Document


INDEX_PATH = "storage/faiss_index.bin"
CHUNKS_PATH = "storage/chunks.json"
MANIFEST_PATH = "storage/index_manifest.json"


def ensure_storage_dir():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)


def save_index(index):
    ensure_storage_dir()
    faiss.write_index(index, INDEX_PATH)


def load_index():
    return faiss.read_index(INDEX_PATH)


def save_chunks(chunks):
    ensure_storage_dir()
    data = [
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        if isinstance(item, str):
            documents.append(Document(page_content=item))
        else:
            documents.append(
                Document(
                    page_content=item["page_content"],
                    metadata=item.get("metadata", {}),
                )
            )
    return documents


def save_manifest(manifest):
    ensure_storage_dir()
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def index_exists(manifest=None):
    return (
        os.path.exists(INDEX_PATH)
        and os.path.exists(CHUNKS_PATH)
        and (manifest is None or load_manifest() == manifest)
    )