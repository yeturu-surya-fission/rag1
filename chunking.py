from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Load PDF and split into chunks with metadata
    """

    # Step 1: Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # Step 3: Split into chunks
    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    file_path = "data/company-policy.pdf"

    chunks = load_and_chunk_pdf(file_path)

    print(f"Total chunks created: {len(chunks)}\n")

    # Print sample chunk
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)
        print("-" * 50)