import os
import time

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunking import load_and_chunk_pdf

load_dotenv(override=True)


def get_setting(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def get_int_setting(name, default):
    try:
        return int(get_setting(name, str(default)))
    except ValueError:
        return default


def create_embeddings(chunks):
    """
    Convert text chunks into embeddings
    """

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")

    model_name = get_setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        api_key=api_key,
    )

    texts = [chunk.page_content for chunk in chunks]
    batch_size = max(1, get_int_setting("EMBEDDING_BATCH_SIZE", 10))
    pause_seconds = max(0.0, float(get_setting("EMBEDDING_BATCH_DELAY", "1.0")))
    max_retries = max(0, get_int_setting("EMBEDDING_MAX_RETRIES", 5))
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(max_retries + 1):
            try:
                vectors.extend(embeddings.embed_documents(batch))
                break
            except Exception:
                if attempt >= max_retries:
                    raise
                time.sleep(min(60.0, 2 ** attempt))

        if start + batch_size < len(texts):
            time.sleep(pause_seconds)

    return vectors


if __name__ == "__main__":
    file_path = "data/company-policy.pdf"

    chunks = load_and_chunk_pdf(file_path)

    vectors = create_embeddings(chunks)

    print(f"Total embeddings created: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")