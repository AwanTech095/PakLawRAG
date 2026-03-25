from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

_SCRIPTS = Path(__file__).parent
load_dotenv(_SCRIPTS / "../.env")
_STORE_PATH = str(_SCRIPTS / "../vectorstore_sections")

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
        )
        _vectorstore = FAISS.load_local(
            _STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def query_vectorstore(query, k=3):
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(query, k=k)

    print(f"\nQuery: {query}")
    print(f"\nTop {k} retrieved sections:\n")

    for i, doc in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Section ID: {doc.metadata.get('section_id')}")
        print(f"Source: {doc.metadata.get('source')}")
        print("-" * 80)
        display = doc.metadata.get("original_text") or doc.page_content
        print(display[:1000])
        print()


if __name__ == "__main__":
    query = input("Enter your legal question: ")
    query_vectorstore(query)