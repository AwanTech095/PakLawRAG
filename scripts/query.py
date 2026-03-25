from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

_SCRIPTS = Path(__file__).parent
_STORE_PATH = str(_SCRIPTS / "../vectorstore_sections")

_vectorstore = None

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant specializing in Pakistani law.
Answer the user's question using ONLY the PPC sections provided below.
Be concise and cite the section numbers in your answer.
If the sections don't contain enough information to answer, say so.

Relevant PPC Sections:
{context}"""),
    ("human", "{question}"),
])


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            _STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def query_vectorstore(query, k=3):
    docs = get_vectorstore().similarity_search(query, k=k)

    context = "\n\n".join(
        f"[Section {d.metadata['section_id']}]\n{d.metadata.get('original_text') or d.page_content}"
        for d in docs
    )

    llm = ChatOllama(model="gemma3:4b", temperature=0)
    chain = PROMPT | llm

    print(f"\nQuery: {query}\n")
    print("=" * 80)
    response = chain.invoke({"context": context, "question": query})
    print(response.content)
    print("=" * 80)
    print("\nSources:", ", ".join(f"§{d.metadata['section_id']}" for d in docs))


if __name__ == "__main__":
    query = input("Enter your legal question: ")
    query_vectorstore(query)
