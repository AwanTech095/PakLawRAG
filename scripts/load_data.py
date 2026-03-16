from langchain_community.document_loaders import PyPDFLoader

def load_docs():
    loader = PyPDFLoader("../data/data_laws.pdf")
    docs = loader.load()
    return docs


if __name__ == "__main__":
    docs = load_docs()
    print(f"Loaded {len(docs)} document pages")
    print(docs[0].page_content[:1000])