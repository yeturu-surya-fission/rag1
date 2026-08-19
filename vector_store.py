import faiss
import numpy as np
from chunking import load_and_chunk_pdf
from embedings import create_embeddings
from embedings import GoogleGenerativeAIEmbeddings


def build_faiss_index(chunks, vectors):
    """
    Build FAISS index from embeddings
    """

    # Convert to numpy array
    vectors = np.array(vectors).astype("float32")

    # Get vector dimension
    dimension = vectors.shape[1]

    # Create FAISS index (Flat L2)
    index = faiss.IndexFlatL2(dimension)

    # Add vectors
    index.add(vectors)

    return index


def search(index, query_vector, chunks, top_k=5):
    """
    Search similar chunks
    """

    query_vector = np.array([query_vector]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


if __name__ == "__main__":
    file_path = "data/company-policy.pdf"

    # Load + chunk
    chunks = load_and_chunk_pdf(file_path)

    # Create embeddings
    vectors = create_embeddings(chunks)

    # Build index
    index = build_faiss_index(chunks, vectors)

    print("FAISS index built successfully!")

    # Example query
    query = "What is maternity leave policy?"

    
    embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    query_vector = embed_model.embed_query(query)

    results = search(index, query_vector, chunks)

    print("\nTop results:\n")
    for r in results:
        print(r.page_content)
        print("-" * 50)