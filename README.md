# PakLawRAG

A semantic search system for the **Pakistan Penal Code (PPC)**. Given a natural language query it returns the most relevant PPC sections using dense vector retrieval.

It is **retrieval-only** — it returns matching sections verbatim, not a generated summary. This is intentional: legal text should be read directly, not paraphrased.

---

## Project structure

```
PakLawRAG/
├── scripts/
│   ├── scrape_ppc.py                 # Fetch and parse all PPC sections from the web
│   ├── normalise_sections.py         # Replace Urdu/Arabic legal terms with English equivalents
│   ├── build_vectorstore_sections.py # Embed sections and build FAISS index
│   ├── query_vectorstore_sections.py # Interactive retrieval CLI
│   └── eval_retrieval.py             # Retrieval evaluation (Hit@k, MRR)
├── output/
│   └── ppc_sections.json             # 636 parsed + normalised sections
├── vectorstore_sections/             # FAISS index (built by build_vectorstore_sections.py)
└── requirements.txt
```

---

## Pipeline

### Build phase (run once)

```
pakistani.org/pakistan/legislation/1860/actXLVof1860.html
      │
      ▼
scrape_ppc.py
      │   ├─ Fetches raw HTML, extracts plain text via BeautifulSoup
      │   ├─ Trims to PPC body (between "Pakistan Penal Code" and last "Schedule")
      │   ├─ Detects section headings with a single compiled regex
      │   │    handles: "375."  "375A."  "337-A."  "120-A"  "Section 375"
      │   ├─ Deduplicates — keeps longest text per section_id
      │   └─ Saves {"section_id", "text"} for 636 sections
      │
      ▼
normalise_sections.py
      │   ├─ Expands Islamized Urdu/Arabic terms inline using single-pass regex
      │   │    e.g. "qatl-i-amd" → "qatl-i-amd (intentional murder, murder, ...)"
      │   ├─ Extracts keywords found per section
      │   └─ Saves {"section_id", "text", "normalized_text", "keywords"}
      │
      ▼
build_vectorstore_sections.py
      │   ├─ Loads sections from ppc_sections.json
      │   ├─ Embeds normalized_text using sentence-transformers/all-MiniLM-L6-v2
      │   ├─ Stores original text in document metadata for display
      │   └─ Builds and saves FAISS flat index
      │
      ▼
vectorstore_sections/
```

### Query phase

```
User question (natural language)
      │
      ▼
query_vectorstore_sections.py
      │   ├─ Loads FAISS index + embedding model
      │   ├─ Embeds the question with the same model
      │   └─ similarity_search(query, k=3) → top-k sections
      │
      ▼
Prints section ID + original legal text for each result
```

### What is embedded vs what is displayed

| Field | Content | Used for |
|---|---|---|
| `normalized_text` | Urdu terms + English meanings in `()` | Embedding / retrieval |
| `text` (original) | Verbatim legal text as scraped | Displayed to user |

The embedding model sees English equivalents of Urdu terms. The user sees clean, unmodified legal text.

---

## Setup

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
cd scripts

# 1. Scrape sections from the web
python scrape_ppc.py

# 2. Normalise Urdu/Arabic terms
python normalise_sections.py

# 3. Build FAISS index
python build_vectorstore_sections.py
```

### Query

```bash
python query_vectorstore_sections.py
```

### Evaluate

```bash
python eval_retrieval.py
```

---

## Design decisions

### Web scraping instead of PDF parsing

The original pipeline loaded the PPC from a PDF using `PyPDFLoader`. This required:
- A local copy of the PDF
- A two-pass anchor-based section extractor (300+ lines)
- A separate cleaning step to strip page markers (`<<<PAGE_N>>>`), footnote refs (`3[Pakistan]`), and omission placeholders (`* * *`)
- `load_data.py`, `parser_data.py`, `split_data.py`, `inspect_data.py`, `clean_and_rebuild.py` — five scripts for one job

The web source at `pakistani.org` provides clean, structured HTML. A single `scrape_ppc.py` replaces all five scripts, produces cleaner text, and has no local file dependency. The only downside is a network request at build time.

The PDF parser also had a hard dependency on `load_data.py` which was an empty stub — meaning it could not run at all.

### Section-level granularity instead of fixed-size chunks

`RecursiveCharacterTextSplitter` (chunk_size=800, overlap=80) was tried first and discarded. A PPC section is the natural retrieval unit. When a user asks "what is the punishment for robbery?" they want Section 392, not an 800-character fragment that may cut off the penalty clause mid-sentence. Section-level chunking also makes `section_id` metadata clean and unambiguous.

### Terminology normalisation

The PPC was Islamized between 1979–1990. Key offence titles were replaced with Arabic/Urdu terms:

| English query term | PPC section title | Sections |
|---|---|---|
| Murder | Qatl-e-Amd | 300, 302 |
| Culpable homicide | Qatl-i-khata | 318, 319 |
| Accidental killing | Qatl-bis-sabab | 321, 322 |
| Blood money | Diyat | 323, 330 |
| Retaliation | Qisas | 304, 307 |
| Head/face wound | Shajjah | 337, 337A |
| Body wound | Jurh | 337B |
| Penetrating wound | Jaifah | 337C, 337D |
| Dismemberment | Itlaf-i-udw | 333, 334 |
| Organ impairment | Itlaf-i-salahiyyat-i-udw | 335, 336 |
| Abortion (early) | Isqat-i-hamal | 338, 338A |
| Abortion (late) | Isqat-i-janin | 338B, 338C |
| Pardon | Afw | 309 |
| Compounding | Sulh | 310 |
| Giving woman as settlement | Badal-i-sulh / Wanni / Swara | 310A |
| Legal heir | Wali | 305 |
| Coercion | Ikrah | 303 |
| Bone-exposing wound | Mudihah | 337, 337A |
| Bone-breaking wound | Hashimah | 337, 337A |
| Bone-displacing wound | Munaqqilah | 337, 337A |
| Brain membrane wound | Damighah | 337, 337A |

A query for "murder" contains no token that appears in Section 300 ("Qatl-e-Amd"). Neither BM25 nor a dense embedding model bridges this without domain knowledge.

`normalise_sections.py` addresses this by annotating Urdu terms inline:
```
"commits qatl-i-amd"  →  "commits qatl-i-amd (intentional murder, murder, ...)"
```

The original Urdu term is preserved (legal accuracy), and the English equivalents are appended so the embedding model has both vocabularies to match against.

**Important limitation:** `normalise_sections.py` is purely dictionary-based. Any Urdu/Arabic term not explicitly listed in `term_map` passes through unchanged. The script has no ability to auto-detect or expand new terms — each one requires a manual entry.

#### Why single-pass regex

The initial implementation used a loop of `re.sub` calls, one per term, longest-first. This caused cascading replacements: after `qatl-i-amd` was expanded, the shorter `qatl` pattern fired again inside the replacement because `-` is a word boundary in regex:

```
"commits qatl-i-amd (intentional murder...)"
          ^^^^
          \bqatl\b fires here → corrupts the text
```

The fix is a single combined `re.compile("term1|term2|...", re.IGNORECASE)` with a callback. Each position in the string is visited exactly once, so no term can match inside a previous replacement.

### What gets embedded vs what gets displayed

`build_vectorstore_sections.py` embeds `normalized_text` — the version with Urdu terms annotated with English equivalents. The original verbatim text is stored in document metadata and shown to the user at query time. This means:

- The embedding model matches against English vocabulary
- The user reads the actual legal text unchanged

Previously `build_vectorstore_sections.py` was embedding `sec["text"]` (original Urdu text), making the normalisation step have no effect on retrieval at all.

### BM25 was tried and rejected

A hybrid BM25 + FAISS approach using Reciprocal Rank Fusion was tested. It made scores worse. Root cause: BM25 is a keyword frequency model. Queries like "Defamation under Pakistan Penal Code" cause BM25 to score sections that contain "Pakistan Penal Code" literally higher than the actual defamation sections — because "Pakistan Penal Code" is three high-frequency terms. Dense-only retrieval is better for this corpus.

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2` — 22M parameters, 384-dimensional embeddings.

Chosen for: fast inference, small memory footprint, no external API, good general-purpose semantic similarity.

Known limitation: not fine-tuned on legal text. The normalisation covers vocabulary gaps but the model still struggles with semantic gaps where correct sections rank just outside top-3. See [Planned improvements](#planned-improvements).

---

## Retrieval performance

Evaluated on 17 hand-labelled queries split into terminology-gap (5) and standard (12) groups. Metrics: **Hit@k** (did a correct section appear in the top-k results?) and **MRR** (mean reciprocal rank — how high up did the correct section appear on average?).

```
  Query                                            Expected        Retrieved@5                        RR  Note
-------------------------------------------------------------------------------------------------------------------
✓  What is the punishment for qatl-e-amd?         300,302         324,316,320,300,302              0.25  terminology gap
✓  Define culpable homicide                       318,319         38,318,103,397,284               0.50  terminology gap
✓  What is accidental killing?                    318,321         80,315,301,321,318               0.25  terminology gap
✓  Blood money compensation for killing           323,330         331,330,323,337X,299             0.50  terminology gap
✓  Hurt and grievous hurt                         332,337L        387,332,397,87,386               0.50  terminology gap
✓  What constitutes theft?                        378,379         382,379,381,413,378              0.50
✓  Punishment for robbery                         390,392         393,392,394,390,398              0.50
✓  Definition of criminal breach of trust         405,406         405,409,408,407,406              1.00
✓  What is the punishment for rape?               375,376         376,375A,496C,175,377B           1.00
✗  Kidnapping or abduction                        359,360,362     367,364,365B,365,369             0.00
✓  Cheating and dishonestly inducing delivery     415,420         415,420,206,421,423              1.00
✓  What is forgery?                               463,465         463,466,469,467,468              1.00
✓  Defamation                                     499,500         499,232,500,376A,509             1.00
✓  Public servant taking bribe                    161             171E,165,161,183,217             0.33
✓  Sedition against the state                     124A            124A,54,224,503,225B             1.00
✓  What is the punishment for dacoity?            391,395         395,400,195,399,412              1.00
✓  Trespass criminal trespass                     441,447         441,442,452,447,451              1.00

--- Summary (all queries) ---
  Hit@1:  47.1%
  Hit@2:  76.5%
  Hit@3:  82.4%
  Hit@5:  94.1%
  MRR:    0.667

--- Terminology-gap queries (5) ---
  Hit@1:  0.0%
  Hit@2:  60.0%
  Hit@3:  60.0%
  Hit@5:  100.0%
  MRR:    0.400

--- Standard queries (12) ---
  Hit@1:  66.7%
  Hit@2:  83.3%
  Hit@3:  91.7%
  Hit@5:  91.7%
  MRR:    0.778
```

**Terminology-gap Hit@5 = 100%** — all five terminology-gap queries find a correct section within the top 5. The gap is in rank — correct sections tend to appear at rank 2–4 rather than rank 1, hence MRR of 0.400 vs 0.778 for standard queries.

**Hit@3 ≠ Hit@5 now** — unlike earlier versions, pushing to k=5 does recover additional correct sections in the terminology-gap group.

---

## What is working

- **636/636 sections scraped** — correctly handles all section ID formats including hyphenated variants (`337-A`, `52-A`, `120-A`)
- **Normalisation is active** — `normalized_text` is what gets embedded; `term_map` covers 35+ Urdu/Arabic terms including all major Islamized offence titles, wound sub-categories, and tribal settlement terms (`wanni`, `swara`, `badal-i-sulh`)
- **Original text preserved** — users see verbatim legal text; the normalised version is internal to the embedding step only
- **Standard English queries** perform well: 91.7% Hit@3, MRR 0.778
- **Terminology-gap queries** all hit within top 5: 100% Hit@5
- **Evaluation harness** covers 17 queries with Hit@1/2/3/5 and MRR, split by query type

---

## What is not working

### 1. Terminology-gap queries rank low (Hit@1 = 0%)

All five terminology-gap queries find the correct section by rank 5, but none rank it first. The normalisation gives the model the right vocabulary, but `all-MiniLM-L6-v2` was not trained for retrieval — it was trained for sentence similarity. A short query like "Define culpable homicide" doesn't score as highly against a long section document as it should.

### 2. Kidnapping fails entirely

"Kidnapping or abduction" retrieves sections 367, 364, 365B — closely related variants — but not the definitional sections 359, 360, 362. This is a pure semantic gap in the model. The correct sections appear at rank 6–8, just outside top-5. Not a terminology issue — no Urdu term is involved.

### 3. `query_vectorstore_sections.py` reloads the model on every query

`load_vectorstore()` is called inside `query_vectorstore()`, meaning the embedding model and FAISS index are loaded from disk on every single query. In an interactive session this adds several seconds per question.

### 4. Hardcoded relative paths in `build_vectorstore_sections.py` and `query_vectorstore_sections.py`

Both use `"../vectorstore_sections"` relative to the working directory. Must be run from `scripts/` or they silently read/write the wrong location. `scrape_ppc.py`, `normalise_sections.py`, and `eval_retrieval.py` all use `Path(__file__).parent` correctly.

---

## Planned improvements

### Better embedding model — Amazon Bedrock

`all-MiniLM-L6-v2` is a general-purpose similarity model, not a retrieval model. The recommended upgrade is **`cohere.embed-multilingual-v3`** via Amazon Bedrock:

```python
from langchain_aws import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    model_id="cohere.embed-multilingual-v3",
    region_name="us-east-1",
    model_kwargs={"input_type": "search_document"}  # use "search_query" at query time
)
```

| | `all-MiniLM-L6-v2` | `cohere.embed-multilingual-v3` |
|---|---|---|
| Parameters | 22M | — |
| Dimensions | 384 | 1024 |
| Languages | English only | 100+ incl. Urdu, Arabic |
| Optimised for | Sentence similarity | Retrieval |
| Urdu queries | ✗ | ✓ |

This addresses two limitations at once: better retrieval quality and Urdu query support. Only `build_vectorstore_sections.py` and `query_vectorstore_sections.py` need to change — the `input_type` parameter must be `search_document` when building the index and `search_query` at retrieval time.

Note: Bedrock model access for Cohere must be explicitly enabled in the AWS console under **Model access** before use.

### Fix model reload on every query

Move vectorstore loading to a module-level singleton in `query_vectorstore_sections.py`:

```python
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISS.load_local(...)
    return _vectorstore
```

---

## Known limitations

1. **Terminology-gap queries rank poorly** — correct sections found but not at rank 1; requires a retrieval-optimised model to fix
2. **Kidnapping semantic gap** — definitional sections rank 6–8, just outside top-5
3. **English queries only** — Urdu script queries not supported with the current model
4. **Dictionary-based normalisation** — any Urdu/Arabic term not in `term_map` is invisible to the embedding model; new terms require manual entries
5. **Static index** — adding new legal documents requires re-running the full build pipeline
6. **Single source** — only the Pakistan Penal Code is indexed; other Pakistani statutes (CrPC, Evidence Act, etc.) are not included
