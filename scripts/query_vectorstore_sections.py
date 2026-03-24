"""
Retrieval interface for the Pakistan Penal Code.
Loads the FAISS index once, then accepts repeated queries.

Usage:
    python query_vectorstore_sections.py
    python query_vectorstore_sections.py -k 10
"""
import argparse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── vectorstore (loaded once) ─────────────────────────────────────────────────

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("Loading vectorstore...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(
            "../vectorstore_sections",
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vectorstore ready.\n")
    return _vectorstore


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, k: int = 5) -> list:
    return get_vectorstore().similarity_search(question, k=k)


def query(question: str, k: int = 5) -> None:
    docs = retrieve(question, k=k)

    print(f"\nQuery: {question}")
    print(f"\nTop {k} retrieved sections:\n")
    for i, doc in enumerate(docs, 1):
        print("=" * 70)
        sid = doc.metadata.get("section_id")
        print(f"  Result {i} | Section {sid}")
        print("-" * 70)
        print(doc.page_content[:800])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the Pakistan Penal Code by natural language query.")
    parser.add_argument("-k", type=int, default=5, help="Number of sections to retrieve (default: 5)")
    args = parser.parse_args()

    while True:
        try:
            question = input("Enter your legal question (or 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in ("exit", "quit", ""):
            break
        query(question, k=args.k)
