"""
Retrieval evaluation for P.

For each test query we know which PPC section(s) should appear in the
top-k results. We compute Hit@k and MRR (Mean Reciprocal Rank).

Run from the scripts/ directory:
    python eval_retrieval.py

NOTE ON TERMINOLOGY GAP:
The Pakistan Penal Code was Islamized in 1979–1990. Many offence titles
were replaced with Arabic/Urdu terms:
  - Murder        → Qatleamd     (Section 300)
  - Culpable homicide → Qatlikhata (Section 319, 320)
  - Hurt / Grievous hurt → now under Section 332 and the 337-series

Queries using English common-law terminology ("murder", "culpable homicide")
cannot be matched by keyword and are genuinely hard for a general-purpose
embedding model. These cases are marked [TERMINOLOGY GAP] below and are
expected to score 0 until the model is fine-tuned or query expansion is added.
"""
from query_vectorstore_sections import retrieve

# ── ground truth ──────────────────────────────────────────────────────────────
# Format: (query, [expected_section_ids], note)
TEST_CASES = [
    # Terminology gap — "murder" maps to Qatleamd (s.300/302) in the PPC
    ("What is the punishment for murder?",          ["300", "302"],          "TERMINOLOGY GAP: murder→Qatleamd"),
    # Terminology gap — "culpable homicide" maps to Qatlikhata (s.319/320)
    ("Define culpable homicide",                    ["319", "320"],          "TERMINOLOGY GAP: culpable homicide→Qatlikhata"),

    ("What constitutes theft?",                     ["378", "379"],          ""),
    ("Punishment for robbery",                      ["390", "392"],          ""),
    ("Definition of criminal breach of trust",      ["405", "406"],          ""),
    ("What is the punishment for rape?",            ["375", "376"],          ""),
    ("Kidnapping or abduction",                     ["359", "360", "362"],   ""),
    ("Cheating and dishonestly inducing delivery",  ["415", "420"],          ""),
    ("What is forgery?",                            ["463", "465"],          ""),
    ("Defamation",                                  ["499", "500"],          ""),

    # Hurt sections: 332 (definition), 337L (punishment for other hurt)
    # 319/320 are now Qatlikhata (accidental killing), not hurt
    ("Hurt and grievous hurt",                      ["332", "337L"],         ""),

    ("Public servant taking bribe",                 ["161"],                 ""),
    ("Sedition against the state",                  ["124A"],                ""),
    ("What is the punishment for dacoity?",         ["391", "395"],          ""),
    ("Trespass criminal trespass",                  ["441", "447"],          ""),
]

K_VALUES = [1, 3, 5]


# ── metrics ───────────────────────────────────────────────────────────────────

def hits_at_k(retrieved_ids: list, expected_ids: list, k: int) -> bool:
    return any(sid in expected_ids for sid in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: list, expected_ids: list) -> float:
    for rank, sid in enumerate(retrieved_ids, 1):
        if sid in expected_ids:
            return 1.0 / rank
    return 0.0


# ── runner ────────────────────────────────────────────────────────────────────

def run_eval(k_max: int = 5) -> None:
    results = []

    for question, expected, note in TEST_CASES:
        docs = retrieve(question, k=k_max)
        retrieved_ids = [doc.metadata.get("section_id", "") for doc in docs]
        rr = reciprocal_rank(retrieved_ids, expected)
        hits = {k: hits_at_k(retrieved_ids, expected, k) for k in K_VALUES}
        results.append((question, expected, retrieved_ids, rr, hits, note))

    print(f"\n{'Query':<50} {'Expected':<15} {'Retrieved@5':<30} {'RR':>5}  Note")
    print("-" * 120)
    for question, expected, retrieved, rr, hits, note in results:
        exp_str  = ",".join(expected)
        ret_str  = ",".join(retrieved[:5])
        hit_mark = "✓" if any(hits.values()) else "✗"
        print(f"{hit_mark} {question[:48]:<48} {exp_str:<15} {ret_str:<30} {rr:>5.2f}  {note}")

    print("\n--- Summary (all queries) ---")
    for k in K_VALUES:
        hit_rate = sum(r[4][k] for r in results) / len(results)
        print(f"  Hit@{k}:  {hit_rate:.1%}")
    mrr = sum(r[3] for r in results) / len(results)
    print(f"  MRR:    {mrr:.3f}")

    # Summary excluding terminology-gap cases
    solvable = [r for r in results if not r[5]]
    if solvable:
        print(f"\n--- Summary (excluding {len(results)-len(solvable)} terminology-gap queries) ---")
        for k in K_VALUES:
            hit_rate = sum(r[4][k] for r in solvable) / len(solvable)
            print(f"  Hit@{k}:  {hit_rate:.1%}")
        mrr = sum(r[3] for r in solvable) / len(solvable)
        print(f"  MRR:    {mrr:.3f}")


if __name__ == "__main__":
    run_eval()
