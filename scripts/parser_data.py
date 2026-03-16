import re
import json
from pathlib import Path
from load_data import load_docs


def clean_text(text: str) -> str:
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def get_full_text():
    docs = load_docs()

    # skip the contents pages
    docs = docs[21:]

    full_text = "\n".join(doc.page_content for doc in docs)
    full_text = clean_text(full_text)

    return full_text


def parse_sections(text: str):

    pattern = r"(?=\n?\s*\d+\.\s)"

    parts = re.split(pattern, text)

    sections = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        match = re.match(r"(\d+)\.\s*(.*)", part, re.DOTALL)

        if match:
            section_id = match.group(1)
            section_text = match.group(0)

            sections.append({
                "section_id": section_id,
                "text": section_text
            })

    return sections


def save_sections(sections):

    Path("../output").mkdir(exist_ok=True)

    with open("../output/ppc_sections.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(sections)} sections.")


if __name__ == "__main__":

    full_text = get_full_text()

    sections = parse_sections(full_text)

    print(f"Total sections parsed: {len(sections)}")

    print("\nExample section:\n")
    print(sections[0]["text"][:1000])

    save_sections(sections)