from load_data import load_docs

def inspect_docs():
    docs = load_docs()

    print(f"Total pages loaded: {len(docs)}")

    # inspect first 3 pages
    for i in range(min(3, len(docs))):
        print("\n" + "="*50)
        print(f"PAGE {i+1}")
        print("="*50)
        print(docs[i].page_content[:1500])

if __name__ == "__main__":
    inspect_docs()