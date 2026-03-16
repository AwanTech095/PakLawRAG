from load_data import load_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs():


    docs = load_docs()
    # remove contents pages (first ~21 pages)
    docs = docs[21:]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":

    chunks = split_docs()
    print(f"Total chunks created: {len(chunks)}")
    print("\nExample chunk:\n")
    print(chunks[0].page_content[:1000])