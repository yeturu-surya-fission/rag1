from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def split_by_chapters(text):
    """
    Split text into chapters using simple regex.
    Works if chapters are like 'CHAPTER 1', 'Chapter 1', etc.
    """
    chapters = re.split(r"(CHAPTER\s+\d+|Chapter\s+\d+)", text)

    # Combine chapter titles with content
    combined = []
    for i in range(1, len(chapters), 2):
        title = chapters[i]
        content = chapters[i + 1] if i + 1 < len(chapters) else ""
        combined.append((title, content))

    return combined


def load_and_chunk_novel(file_path: str):
    """
    Load PDF novel and chunk using:
    - Chapter-aware splitting
    - Paragraph-first recursive chunking
    - High overlap for context preservation
    """

    # Step 1: Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Combine all pages into one text
    full_text = "\n".join([doc.page_content for doc in documents])

    # Step 3: Split into chapters (if present)
    chapters = split_by_chapters(full_text)

    # Step 4: Recursive splitter (paragraph → sentence → word)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []

    # Step 5: Process each chapter
    for chapter_title, chapter_text in chapters:
        chunks = splitter.create_documents([chapter_text])

        for chunk in chunks:
            chunk.metadata["chapter"] = chapter_title.strip()
            chunk.metadata["source"] = file_path

        all_chunks.extend(chunks)

    return all_chunks


def load_and_chunk_pdf(file_path: str):
    """Load a PDF and split its pages into searchable, metadata-rich chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for document in documents:
        page_chunks = splitter.split_documents([document])
        for chunk in page_chunks:
            chunk.metadata["source"] = file_path
        chunks.extend(page_chunks)

    return chunks


if __name__ == "__main__":
    file_path = "data/novel.pdf"

    chunks = load_and_chunk_novel(file_path)

    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}:")
        print(chunk.page_content[:300])
        print("Metadata:", chunk.metadata)
        print("-" * 50)