# PakLawRAG

A semantic search system for the **Pakistan Penal Code (PPC)**. Given a natural language query it returns the most relevant PPC sections using dense vector retrieval.

---

## What it does

The system parses the full PPC from a PDF, extracts all 636 individual sections, embeds them using a sentence-transformer model, indexes them in FAISS, and exposes an interactive CLI for natural language search.

It is **retrieval-only** — it returns matching sections directly, not a generated summary. This is intentional: legal text should be read verbatim, not paraphrased.

---

## Project structure

```
PakLawRAG/
├── scripts/
│   ├── load_data.py                  # Load PDF pages via PyPDFLoader
│   ├── inspect_data.py               # Quick inspection of raw PDF output
│   ├── parser_data.py                # Core section extraction from raw text
│   ├── split_data.py                 # Alternative chunk-based splitter (unused in main pipeline)
│   ├── build_vectorstore_sections.py # Build FAISS index from ppc_sections.json
│   ├── clean_and_rebuild.py          # Clean artifacts in JSON then rebuild vectorstore
│   ├── query_vectorstore_sections.py # Interactive retrieval CLI
│   └── eval_retrieval.py             # Retrieval evaluation (Hit@k, MRR)
├── output/
│   ├── ppc_sections.json             # 636 parsed + cleaned sections
│   ├── expected_section_ids.txt      # 637 section IDs found in the table of contents
│   ├── found_section_ids.txt         # 636 successfully extracted IDs
│   └── missing_section_ids.txt       # 1 missing section (108A)
├── vectorstore_sections/
│   └── index.pkl                     # Serialised FAISS index + embeddings
└── requirements.txt
```

---

## Pipeline

The system has two phases: **build** (run once) and **query** (run any time).

### Build phase

```
data/data_laws.pdf
      │
      ▼
load_data.py          PyPDFLoader → ~600 LangChain Document objects (one per page)
      │
      ▼
parser_data.py        Section extraction
      │               ├─ Pages 0–20  → table of contents → extract expected section IDs (637)
      │               ├─ Pages 21+   → body text with <<<PAGE_N>>> markers for boundary tracking
      │               ├─ First pass  → find each section heading in reading order
      │               ├─ Rescue pass → retry any missed sections without cursor constraint
      │               ├─ Build text blocks between consecutive anchors
      │               └─ Normalise + deduplicate → 636 sections
      │
      ▼
output/ppc_sections.json    [{"section_id": "300", "text": "300. Murder..."}, ...]
      │
      ▼
clean_and_rebuild.py  Post-processing pass (see Data cleaning below)
      │               └─ Overwrites ppc_sections.json with clean text
      │
      ▼
build_vectorstore_sections.py
      │               ├─ Each section → LangChain Document
      │               │    page_content = section text
      │               │    metadata     = {section_id, source}
      │               ├─ Embed with sentence-transformers/all-MiniLM-L6-v2
      │               └─ Build + serialise FAISS flat index
      │
      ▼
vectorstore_sections/index.pkl
```

### Query phase

```
User question (natural language)
      │
      ▼
query_vectorstore_sections.py
      │   ├─ Load FAISS index once (singleton — not reloaded per query)
      │   ├─ Embed the question with the same model
      │   └─ similarity_search(query, k=5) → top-k sections by cosine similarity
      │
      ▼
Print section ID + full section text for each result
```

---

## Key design decisions

### Section-level granularity instead of fixed-size chunks

`split_data.py` exists and implements a standard `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=80). It was tried first but discarded in favour of section-level documents.

**Why:** A PPC section is the natural retrieval unit. When a user asks "what is the punishment for robbery?" they want Section 392, not a 800-character fragment that might cut off the penalty clause mid-sentence. Section-level chunking also makes metadata (section_id) clean and unambiguous.

The chunk-based script is kept for reference but is not part of the active pipeline.

### Anchor-based section extraction

The parser uses a two-pass anchor approach rather than splitting on regex alone.

- **First pass** — scans the body text in reading order, advancing a cursor after each match to avoid false positives from cross-references (e.g. "see section 300" later in the document).
- **Rescue pass** — any section not found in the first pass is retried without the cursor, since a small number of sections appear out of order due to PDF formatting.
- **Deduplication** — if both passes find the same section, the longer text block is kept.

This achieves 636/637 sections (99.8%). The only missing section is **108A**, which does not appear in the body text of this specific PDF edition.

### Data cleaning

The raw PDF → text conversion introduces several noise classes. `clean_and_rebuild.py` (and the same logic in `parser_data.py`'s `clean_section_text`) removes them:

| Noise type | Example | Fix |
|---|---|---|
| Page break markers | `<<<PAGE_42>>>` | Stripped by regex |
| Inline footnote refs | `3[Pakistan]` | Unwrapped: `→ Pakistan` (5 passes to handle nesting like `2[3[text]]`) |
| Omission placeholders | `* * * * * *` | Removed |
| Footnote citation lines | `1. Ins. by Act...`, `2Subs.` | Lines matching citation patterns dropped |

These artifacts end up in the section text during parsing because page markers are injected intentionally (to help boundary detection) and footnote refs are part of the raw PDF text. Cleaning them before embedding improves retrieval quality because the model can focus on legal semantics rather than noise tokens.

### Vectorstore loaded once

The original `query_vectorstore_sections.py` called `load_local()` inside the query function, meaning the 500 KB index and the sentence-transformer model were loaded from disk on **every query**. The fixed version uses a module-level singleton (`_vectorstore`): the model and index are loaded on the first query and reused for all subsequent ones in the same session.

### Embedding model

**`sentence-transformers/all-MiniLM-L6-v2`** — a 22M parameter model producing 384-dimensional embeddings.

Chosen because:
- Fast inference, small memory footprint
- Good general-purpose semantic similarity
- No external API required — runs fully locally

Known limitation: it is not fine-tuned on legal text. Queries using English common-law terminology that maps to Islamized Urdu terms in the PPC will underperform. A model fine-tuned on Pakistani legal text would improve accuracy significantly.

### Why BM25 hybrid was tried and rejected

During development a hybrid BM25 + FAISS approach was implemented using Reciprocal Rank Fusion. It made scores **worse** (Hit@5 dropped from 66.7% to 60%). Root cause:

- BM25 is a keyword frequency model. Queries like `"Defamation under Pakistan Penal Code"` cause BM25 to score sections that contain the phrase "Pakistan Penal Code" literally (e.g. Section 402D) higher than the actual defamation sections — because "defamation" is one term while "Pakistan Penal Code" is three high-frequency terms.
- The dense model handles semantic similarity better for this corpus. BM25 only adds noise.

---

## Retrieval performance

Evaluated on 15 hand-labelled queries across key PPC topics. Metrics: **Hit@k** (did a correct section appear in top-k results?) and **MRR** (mean reciprocal rank).

```
Query                                              Expected        Retrieved@5                       RR  Note
------------------------------------------------------------------------------------------------------------------------
✗ What is the punishment for murder?               300,302         396,506,115,325,57              0.00  TERMINOLOGY GAP: murder→Qatleamd
✗ Define culpable homicide                         319,320         284,38,397,87,440               0.00  TERMINOLOGY GAP: culpable homicide→Qatlikhata
✓ What constitutes theft?                          378,379         382,381,379,381A,378            0.33
✓ Punishment for robbery                           390,392         392,393,394,390,397             1.00
✓ Definition of criminal breach of trust           405,406         405,409,407,408,406             1.00
✓ What is the punishment for rape?                 375,376         376,496C,506,292A,377B          1.00
✗ Kidnapping or abduction                          359,360,362     367,367A,369,364,364A           0.00
✓ Cheating and dishonestly inducing delivery       415,420         415,420,418,416,421             1.00
✓ What is forgery?                                 463,465         463,469,468,466,467             1.00
✓ Defamation                                       499,500         499,500,509,503,376A            1.00
✗ Hurt and grievous hurt                           332,337L        387,87,397,337J,386             0.00
✓ Public servant taking bribe                      161             162,171E,165,161,217            0.25
✓ Sedition against the state                       124A            124A,503,207,225B,120B          1.00
✓ What is the punishment for dacoity?              391,395         395,396,195,400,399             1.00
✓ Trespass criminal trespass                       441,447         441,442,447,449,443             1.00

--- Summary (all 15 queries) ---
  Hit@1:  60.0%
  Hit@3:  66.7%
  Hit@5:  73.3%
  MRR:    0.639

--- Summary (excluding 2 terminology-gap queries) ---
  Hit@1:  69.2%
  Hit@3:  76.9%
  Hit@5:  84.6%
  MRR:    0.737
```

**Terminology gap — the most important limitation:**

The Pakistan Penal Code was Islamized between 1979–1990. Key offence titles were replaced with Arabic/Urdu terms:

| English query term | PPC section title | Section |
|---|---|---|
| Murder | Qatleamd | 300, 302 |
| Culpable homicide | Qatlikhata | 319, 320 |
| Hurt / Grievous hurt | Hurt (Islamized framework) | 332, 337-series |

A query for "murder" contains no token that appears in Section 300 ("Qatleamd"). Neither BM25 nor the dense model bridges this without either (a) query expansion mapping English terms to their Urdu equivalents, or (b) fine-tuning the model on PPC text.

**Kidnapping** fails because the retrieved sections (367, 367A) are closely related kidnapping variants but not the precise definitional sections (359, 360, 362). Increasing k retrieves them — they appear at rank 6–8.

**Hurt/grievous hurt** — the original eval had wrong expected sections (319/320 are Qatlikhata, not hurt). The corrected expected sections are 332 (Hurt) and 337L (Punishment for other hurt). Even with corrections the model returns extortion/robbery sections that share "bodily harm" vocabulary.

**To improve scores:** add query expansion with a mapping of English common-law terms to PPC Urdu equivalents, or fine-tune the embedding model on PPC text.

---

## Setup

### Requirements

```
Python 3.10+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages used:
- `langchain-community` — PDF loader, FAISS wrapper, HuggingFace embeddings
- `langchain-core` — Document objects, pipeline primitives
- `faiss-cpu` — vector similarity index
- `sentence-transformers` — embedding model (`all-MiniLM-L6-v2`)
- `pypdf` — PDF text extraction

### First-time setup (if starting from the PDF)

Place the PPC PDF at `data/data_laws.pdf`, then run the pipeline in order:

```bash
cd scripts

# 1. Parse sections from PDF → output/ppc_sections.json
python parser_data.py

# 2. Clean artifacts + build FAISS index
python clean_and_rebuild.py
```

### If ppc_sections.json already exists

The cleaned JSON and vectorstore are already committed. You only need to run `clean_and_rebuild.py` again if you re-parse from the PDF.

---

## Usage

### Interactive search

```bash
cd scripts
python query_vectorstore_sections.py
```

```
Loading vectorstore...
Vectorstore ready.

Enter your legal question (or 'exit'): What is the punishment for dacoity?

Query: What is the punishment for dacoity?

Top 5 retrieved sections:

======================================================================
  Result 1 | Section 395
----------------------------------------------------------------------
395. Punishment for dacoity. Whoever commits dacoity shall be punished with
imprisonment for life, or with rigorous imprisonment for a term which may
extend to ten years, and shall also be liable to fine.
...
```

**Options:**

```bash
python query_vectorstore_sections.py -k 10   # retrieve top 10 instead of 5
```

### Evaluation

```bash
cd scripts
python eval_retrieval.py
```

Runs 15 ground-truth queries and prints Hit@1/3/5 and MRR.

---

## Planned improvements

Current Hit@5 is 73.3% overall (84.6% excluding terminology-gap queries). Getting to 90%+ requires two separate fixes targeting two separate problems.

### Fix 1 — Query expansion (for the terminology gap)

No embedding model, however capable, knows that "murder" maps to "Qatleamd" in Pakistani law. That mapping is domain-specific knowledge that doesn't exist in any general pre-training corpus. A better model will not solve this.

The right tool is **query expansion**: before the query reaches the vectorstore, rewrite it using a dictionary of English common-law terms → PPC Urdu/Arabic equivalents.

```
"What is the punishment for murder?"
    → expanded to include "Qatleamd"
    → now matches Section 300 trivially
```

The dictionary only needs to cover the Islamized terms. Everything else already works.

### Fix 2 — Better embedding model (for semantic mismatches)

`all-MiniLM-L6-v2` is a 22M parameter model optimised for speed, not retrieval accuracy. It is the reason "hurt" retrieves extortion sections and "kidnapping" ranks the definitional sections at position 6–8 instead of 1–5.

`BAAI/bge-base-en-v1.5` is a dedicated retrieval model that consistently outperforms MiniLM-class models on the BEIR benchmark. It is 438 MB vs MiniLM's 80 MB — larger but still runs fully locally with no API.

One implementation detail: BGE models expect queries to be prefixed with `"Represent this sentence for searching relevant passages: "` at search time (not at index time). The documents are embedded as-is; only the query changes.

Switching requires rebuilding the vectorstore once with the new model. After that, the query interface is identical.

---

## Known limitations

1. **Terminology gap (Islamization)** — Murder, culpable homicide, and hurt were renamed in the 1979–1990 Islamization amendments. English common-law queries cannot be matched against these sections without query expansion or domain fine-tuning.
2. **Missing section 108A** — not present in this PDF edition.
3. **Generic embedding model** — `all-MiniLM-L6-v2` has no legal domain fine-tuning. Achieves Hit@5 of 84.6% on non-terminology-gap queries.
4. **BM25 does not help** — keyword search was tested and made results worse because common domain phrases ("Pakistan Penal Code") swamp the keyword signal. Dense-only is better for this corpus.
5. **English only** — the PPC body text is in English; Urdu queries are not supported.
6. **Static index** — adding new legal documents requires running `clean_and_rebuild.py` again.
