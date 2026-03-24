"""
Clean ppc_sections.json of artifacts, then rebuild the FAISS vectorstore.
Run once from the scripts/ directory:  python clean_and_rebuild.py
"""
import re
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean_section_text(text: str) -> str:
    # Remove <<<PAGE_N>>> markers
    text = re.sub(r"\s*<<<PAGE_\d+>>>\s*", " ", text)

    # Unwrap inline footnote references: 3[Pakistan] → Pakistan (handle nesting)
    for _ in range(5):
        text = re.sub(r"\d+\[([^\[\]]*)\]", r"\1", text)

    # Remove omission placeholders: * * * * *
    text = re.sub(r"(\*\s*){2,}", "", text)

    # Remove footnote citation lines like "1. Ins. by Act..." or "2Subs."
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        s = line.strip()
        if re.match(r"^\d+[\. ]+(Ins|Sub|Rep|Omit|Added|Amended|Renumber|See|Vide)\b", s):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_sections(sections: list) -> list:
    return [
        {"section_id": sec["section_id"], "text": clean_section_text(sec["text"])}
        for sec in sections
    ]


# ── vectorstore ───────────────────────────────────────────────────────────────

def build_vectorstore(sections: list):
    documents = [
        Document(
            page_content=sec["text"],
            metadata={"section_id": sec["section_id"], "source": f"PPC Section {sec['section_id']}"}
        )
        for sec in sections
    ]

    print(f"Building embeddings for {len(documents)} sections...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    save_path = "../vectorstore_sections"
    Path(save_path).mkdir(exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Vectorstore saved to {save_path}")
    return vectorstore


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    json_path = "../output/ppc_sections.json"
    with open(json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)
    print(f"Loaded {len(sections)} sections")

    sections = clean_sections(sections)

    # Show a before/after sample
    sample = sections[2]  # Section 3 had PAGE markers
    print(f"\nSample (Section {sample['section_id']}):\n{sample['text'][:300]}\n")

    # Save cleaned JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    print("Saved cleaned ppc_sections.json")

    build_vectorstore(sections)
    print("\nDone.")
