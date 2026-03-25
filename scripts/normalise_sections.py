import json
import re
from pathlib import Path

_SCRIPTS = Path(__file__).parent

term_map = {
    "qatl-i-amd": [
        "intentional murder",
        "intentional killing",
        "murder",
        "wilful murder",
        "homicide"
    ],
    "qatl i amd": [
        "intentional murder",
        "intentional killing",
        "murder",
        "wilful murder",
        "homicide"
    ],
    "qatl-e-amd": [
        "intentional murder",
        "intentional killing",
        "murder",
        "wilful murder",
        "homicide"
    ],
    "qatl": [
        "killing",
        "homicide"
    ],
    "qatl shibh-i-amd": [
        "quasi intentional homicide",
        "similar to intentional killing",
        "semi intentional killing"
    ],
    "qatl shibh i amd": [
        "quasi intentional homicide",
        "similar to intentional killing",
        "semi intentional killing"
    ],
    "qatl-e-shibh-i-amd": [
        "quasi intentional homicide",
        "similar to intentional killing",
        "semi intentional killing"
    ],
    "qatl-i-khata": [
        "accidental killing",
        "unintentional killing",
        "homicide by mistake",
        "manslaughter by mistake"
    ],
    "qatl i khata": [
        "accidental killing",
        "unintentional killing",
        "homicide by mistake",
        "manslaughter by mistake"
    ],
    "qatl-e-khata": [
        "accidental killing",
        "unintentional killing",
        "homicide by mistake",
        "manslaughter by mistake"
    ],
    "diyat": [
        "blood money",
        "financial compensation",
        "compensation to legal heirs"
    ],
    "ta'zir": [
        "discretionary punishment",
        "judge determined punishment"
    ],
    "tazir": [
        "discretionary punishment",
        "judge determined punishment"
    ],
    "fasad-fil-arz": [
        "grave societal harm",
        "mischief on earth"
    ],
    "fasad fil arz": [
        "grave societal harm",
        "mischief on earth"
    ],
    "arsh": [
        "specified compensation",
        "legal compensation"
    ],
    "daman": [
        "compensation for hurt",
        "damages"
    ],
    "zirh": [
        "hurt",
        "injury"
    ],
    "itlaf-i-udw": [
        "destruction of organ",
        "loss of organ"
    ],
    "itlaf-i-salahiyyat-i-udw": [
        "impairment of organ function",
        "loss of function of organ"
    ],
    "jurh": [
        "wound",
        "injury"
    ]
}

def clean_text(text):
    # Fix 2: preserve newlines — only collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def find_keywords(text, term_map):
    found = []
    seen = set()  # Fix minor: O(1) deduplication
    lower_text = text.lower()
    for term, meanings in term_map.items():
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, lower_text):
            for item in [term] + meanings:
                if item not in seen:
                    seen.add(item)
                    found.append(item)
    return found

def make_normalized_text(text, term_map):
    # Fix 1: single-pass replacement so shorter terms can't re-match
    # inside already-replaced text (e.g. "qatl" firing inside "qatl-i-amd (...)").
    sorted_terms = sorted(term_map.items(), key=lambda x: len(x[0]), reverse=True)

    combined = re.compile(
        "|".join(r"\b" + re.escape(term) + r"\b" for term, _ in sorted_terms),
        re.IGNORECASE,
    )
    lookup = {term.lower(): meanings for term, meanings in sorted_terms}

    def replace(m):
        matched = m.group(0)
        meanings = lookup.get(matched.lower(), [])
        return matched + " (" + ", ".join(meanings) + ")"

    normalized = combined.sub(replace, text)
    normalized = clean_text(normalized)
    return normalized

def main():
    # Fix 3: use __file__-relative paths so script works from any directory.
    # Write back to ppc_sections.json so build_vectorstore_sections.py picks it up.
    input_file  = _SCRIPTS / "../output/ppc_sections.json"
    output_file = _SCRIPTS / "../output/ppc_sections.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for item in data:
        section_id = str(item.get("section_id", "")).strip()
        # support both old {"text"} and new {"original_text"} schemas
        text = (item.get("original_text") or item.get("text", "")).strip()

        normalized_text = make_normalized_text(text, term_map)
        keywords = find_keywords(text, term_map)

        new_item = {
            "section_id":      section_id,
            "text":            text,
            "normalized_text": normalized_text,
            "keywords":        keywords,
        }

        new_data.append(new_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"done. saved to {output_file}")
    print(f"total sections processed: {len(new_data)}")

if __name__ == "__main__":
    main()