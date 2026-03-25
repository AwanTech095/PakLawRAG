import json
import re

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
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_keywords(text, term_map):
    found = []
    lower_text = text.lower()
    for term, meanings in term_map.items():
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, lower_text):
            found.append(term)
            for m in meanings:
                found.append(m)
    seen = []
    for item in found:
        if item not in seen:
            seen.append(item)
    return seen

def make_normalized_text(text, term_map):
    normalized = text

    for term, meanings in sorted(term_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"\b" + re.escape(term) + r"\b"

        replacement = term + " (" + ", ".join(meanings) + ")"

        normalized = re.sub(
            pattern,
            replacement,
            normalized,
            flags=re.IGNORECASE
        )

    normalized = clean_text(normalized)
    return normalized

def main():
    input_file = "../output/ppc_sections.json"
    output_file = "../output/ppc_sections_normalized.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for item in data:
        section_id = str(item.get("section_id", "")).strip()
        text = item.get("text", "").strip()

        normalized_text = make_normalized_text(text, term_map)
        keywords = find_keywords(text, term_map)

        new_item = {
            "section_id": section_id,
            "text": text,
            "normalized_text": normalized_text,
            "keywords": keywords
        }

        new_data.append(new_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"done. saved to {output_file}")
    print(f"total sections processed: {len(new_data)}")

if __name__ == "__main__":
    main()