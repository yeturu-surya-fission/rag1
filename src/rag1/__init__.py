import sys
import os

def main() -> None:
    # Add root directory to sys.path if not present
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from ragpipeline import setup, generate_answer

    index, chunks, bm25 = setup()
    print("RAG Ready (No Cache Mode)\n")

    while True:
        try:
            q = input("Ask: ")
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() == "exit":
            break

        ans = generate_answer(q, index, chunks, bm25)
        print("\nAnswer:", ans)
        print("-" * 50)

