import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from chunking import load_and_chunk_pdf

load_dotenv()


def get_setting(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


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

    vectors = embeddings.embed_documents(texts)

    return vectors


if __name__ == "__main__":
    file_path = "data/company-policy.pdf"

    chunks = load_and_chunk_pdf(file_path)

    vectors = create_embeddings(chunks)

    print(f"Total embeddings created: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")