# PakLawRAG

A RAG (Retrieval-Augmented Generation) system for the **Pakistan Penal Code (PPC)**. Given a natural language legal query, it retrieves the most relevant PPC sections and generates a concise answer with section citations.

The core system remains the same in local and deployed modes:

```
User query -> FAISS retrieval over PPC sections -> retrieved evidence -> LLM answer with citations
```

Locally, the system can use `gemma3:4b` through Ollama. On Streamlit Cloud, local Ollama models are not available, so the deployed app uses a hosted Groq model for answer generation while keeping the same FAISS retriever, corpus, embeddings, and evidence display.

---

## Project structure

```
PakLawRAG/
|-- app.py                            # Streamlit web app
|-- scripts/
|   |-- scrape_ppc.py                 # Fetch and parse PPC sections from the web
|   |-- normalise_sections.py         # Add English meanings for Urdu/Arabic legal terms
|   |-- build_vectorstore_sections.py # Embed sections and build FAISS index
|   |-- query.py                      # Local CLI RAG query script
|   |-- eval_retrieval.py             # Retrieval evaluation (Hit@k, MRR)
|   |-- eval_answers.py               # Strict answer checklist evaluation
|   `-- eval_llm_judge.py             # LLM-as-judge groundedness evaluation
|-- output/
|   |-- ppc_sections.json             # 636 parsed + normalised sections
|   |-- answer_eval_report.json       # Strict answer evaluation report
|   `-- llm_judge_eval_report.json    # LLM judge evaluation report
|-- vectorstore_sections/             # Committed FAISS index
|   |-- index.faiss                   # Embedding vectors
|   `-- index.pkl                     # Document metadata
`-- requirements.txt
```

---

## Pipeline

### Build phase (run once)

```
pakistani.org/pakistan/legislation/1860/actXLVof1860.html
      |
      v
scrape_ppc.py
      |   |-- Fetches raw HTML, extracts plain text with BeautifulSoup
      |   |-- Trims to PPC body
      |   |-- Detects section headings such as "375.", "375A.", "337-A.", "120-A"
      |   |-- Deduplicates by keeping the longest text per section_id
      |   `-- Saves {"section_id", "text"} records
      |
      v
normalise_sections.py
      |   |-- Expands Urdu/Arabic legal terms inline with English meanings
      |   |    example: "qatl-i-amd" -> "qatl-i-amd (intentional murder, murder, ...)"
      |   |-- Extracts matched keywords per section
      |   `-- Saves {"section_id", "text", "normalized_text", "keywords"}
      |
      v
build_vectorstore_sections.py
      |   |-- Loads sections from ppc_sections.json
      |   |-- Embeds normalized_text using BAAI/bge-large-en-v1.5
      |   |-- Stores original text in document metadata for display
      |   `-- Builds and saves FAISS index
      |
      v
vectorstore_sections/
```

### Query phase

```
User question
      |
      v
Retriever
      |   |-- Loads FAISS index + BGE embedding model
      |   |-- Embeds the user query
      |   `-- similarity_search(query, k=3) -> top PPC sections
      |
      v
LLM
      |   |-- Receives retrieved PPC sections as context
      |   |-- Answers using only the retrieved evidence
      |   `-- Cites section numbers
      |
      v
Answer + retrieved evidence
```

### What is embedded vs what is displayed

| Field | Content | Used for |
|---|---|---|
| `normalized_text` | PPC text plus English meanings for Urdu/Arabic legal terms | Embedding / retrieval |
| `text` | Original legal text as scraped | Displayed as evidence and passed to the LLM |

The embedding model sees both PPC terminology and English equivalents. The LLM receives the actual legal text as evidence.

---

## Setup

```bash
pip install -r requirements.txt
```

For local Ollama-based scripts:

```bash
ollama pull gemma3:4b
```

For the Streamlit deployment, add a Groq API key in Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_key_here"
LLM_MODEL = "llama-3.1-8b-instant"
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

> The `vectorstore_sections/` directory is committed to the repo, so rebuilding is only needed after changing the corpus or normalization.

### Query locally

```bash
python scripts/query.py
```

### Run Streamlit locally

```bash
streamlit run app.py
```

### Deploy on Streamlit Cloud

1. Push the repo to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select the repo, branch, and `app.py` path.
4. Add `GROQ_API_KEY` in Streamlit Secrets.
5. Deploy.

Streamlit Cloud cannot access a local Ollama server or local model files such as `gemma3:4b`. For this reason, the deployed app uses `ChatGroq` with `llama-3.1-8b-instant` by default. Retrieval, vector store, evidence display, prompts, and evaluation setup remain the same.

---

## Design decisions

### Web scraping instead of PDF parsing

The original pipeline loaded the PPC from a PDF. This required:

- A local PDF file
- A complex anchor-based section extractor
- Extra cleaning for page markers, footnote markers, and omission placeholders
- Multiple scripts for loading, parsing, splitting, inspection, and rebuilding

The web source at `pakistani.org` provides cleaner structured HTML. A single `scrape_ppc.py` replaces that older multi-step PDF pipeline. The tradeoff is that scraping requires network access during rebuild.

### Section-level granularity instead of fixed-size chunks

PPC sections are the natural retrieval unit. A fixed-size chunk can cut a punishment clause or definition in the middle. Section-level retrieval keeps each answer grounded in complete legal provisions and makes citation metadata clean.

### Terminology normalisation

The PPC uses Urdu/Arabic legal terms for several offences and remedies. Users often ask in plain English.

| English query term | PPC term | Sections |
|---|---|---|
| Murder | Qatl-e-Amd / Qatl-i-Amd | 300, 302 |
| Culpable homicide | Qatl-i-khata | 318, 319 |
| Accidental killing | Qatl-bis-sabab | 321, 322 |
| Blood money | Diyat | 323, 330 |
| Retaliation | Qisas | 304, 307 |
| Head/face wound | Shajjah | 337, 337A |
| Body wound | Jurh | 337B |
| Penetrating wound | Jaifah | 337C, 337D |
| Dismemberment | Itlaf-i-udw | 333, 334 |
| Organ impairment | Itlaf-i-salahiyyat-i-udw | 335, 336 |
| Abortion | Isqat-i-hamal / Isqat-i-janin | 338-338C |
| Compromise / settlement | Sulh | 310 |
| Legal heir | Wali | 305 |
| Coercion | Ikrah | 303 |

A query for "murder" may not strongly match Section 300 or 302 if the corpus only says "qatl-e-amd". `normalise_sections.py` solves this by adding English meanings inline:

```
"commits qatl-i-amd" -> "commits qatl-i-amd (intentional murder, murder, ...)"
```

The original term is preserved for legal accuracy, while English equivalents improve retrieval.

**Important limitation:** normalisation is dictionary-based. New Urdu/Arabic terms must be added manually to `term_map`.

### Why single-pass regex

The first approach used repeated `re.sub` calls, one term at a time. That caused cascading replacements, where a shorter term such as `qatl` could match inside a longer replacement such as `qatl-i-amd (...)`.

The current approach builds one combined regex and uses a callback. Each text position is visited once, so replacements do not re-trigger inside already expanded text.

### Embedding model

`BAAI/bge-large-en-v1.5` is used for embeddings.

- 1024-dimensional dense embeddings
- Retrieval-oriented
- Runs locally
- No API key required

It was chosen over `all-MiniLM-L6-v2` because it improved retrieval:

| Metric | MiniLM-L6-v2 | BGE-large-en-v1.5 |
|---|---:|---:|
| Hit@1 | 47.1% | **64.7%** |
| Hit@3 | 82.4% | **94.1%** |
| Hit@5 | 94.1% | **94.1%** |
| MRR | 0.667 | **0.794** |
| Terminology-gap MRR | 0.400 | **0.700** |

### LLMs

Local development:

```
gemma3:4b via Ollama
```

Streamlit Cloud deployment:

```
llama-3.1-8b-instant via Groq
```

The deployed model changed only because Streamlit Cloud cannot run or access the local Ollama model. The RAG architecture remains unchanged: retrieved PPC sections are still passed as context, and the answer is instructed to cite section numbers.

### BM25 was tried and rejected

A hybrid BM25 + FAISS approach using Reciprocal Rank Fusion was tested. It made retrieval worse. BM25 over-weighted generic phrases such as "Pakistan Penal Code" and sometimes ranked irrelevant sections higher than the actual offence sections. Dense-only retrieval performed better for this corpus.

---

## Evaluation

PakLawRAG is evaluated at three levels:

1. **Retrieval evaluation**: whether the correct PPC section appears in top-k results.
2. **Strict answer evaluation**: whether the generated answer includes required facts and citations.
3. **LLM-as-judge groundedness evaluation**: whether the answer is supported by retrieved evidence and avoids contradictions.

### Retrieval performance

Evaluated on 17 hand-labelled queries split into terminology-gap and standard English queries.

Metrics:

- **Hit@k**: whether a correct section appears in the top-k retrieved results
- **MRR**: mean reciprocal rank of the first correct result

```
--- Summary (all queries) ---
  Hit@1:  64.7%
  Hit@2:  94.1%
  Hit@3:  94.1%
  Hit@5:  94.1%
  MRR:    0.794

--- Terminology-gap queries (5) ---
  Hit@1:  40.0%
  Hit@2:  100.0%
  Hit@3:  100.0%
  Hit@5:  100.0%
  MRR:    0.700

--- Standard queries (12) ---
  Hit@1:  75.0%
  Hit@2:  91.7%
  Hit@3:  91.7%
  Hit@5:  91.7%
  MRR:    0.833
```

Interpretation:

- Top-3 retrieval is strong overall: **94.1% Hit@3**.
- Terminology normalisation is useful: terminology-gap queries reach **100% Hit@3**.
- Some exact-rank errors remain, especially for short or ambiguous queries.

### Strict answer evaluation

Script:

```bash
python scripts/eval_answers.py
```

Report:

```text
output/answer_eval_report.json
```

This evaluation uses a stricter checklist method. Each test case defines:

- expected section citations
- required legal facts
- forbidden or contradictory claims

The current selected evaluation set contains 10 cases. It is intentionally strict and includes one known weak case: **public servant taking bribe**, where retrieval tends to miss Section 161 and retrieve related bribery/corruption provisions instead.

```
Strict answer evaluation:
  Total cases: 10
  Passed:      9
  Failed:      1
  Accuracy:    90.0%
```

Interpretation:

- This is a high-precision criterion.
- A failure does not always mean the answer is completely irrelevant.
- It means the answer missed a required section, required fact, or citation under the checklist.
- The known failure is still legally related, but it misses the expected Section 161 under the strict rubric.

### LLM-as-judge groundedness evaluation

Script:

```bash
python scripts/eval_llm_judge.py
```

Report:

```text
output/llm_judge_eval_report.json
```

This evaluation uses a second local LLM as a neutral judge:

```
Answer model: gemma3:4b
Judge model:  llama3.2:latest
```

The judge receives:

- the user question
- retrieved PPC evidence
- generated answer
- expected section IDs, when available

It checks whether the answer:

- contradicts the retrieved evidence
- makes unsupported major claims
- misses the central answer when evidence contains it
- uses citations correctly

Current 10-case groundedness result:

```
LLM judge groundedness:
  Total cases: 10
  Passed:      10
  Failed:      0
  Accuracy:    100.0%
```

Interpretation:

- The judge evaluation is less strict than the checklist evaluation.
- It focuses on whether the answer is grounded and non-contradictory.
- This explains why it may pass cases where the strict checklist fails.
- For example, if a bribery query retrieves related bribery/corruption sections but misses Section 161, the answer may still be grounded in retrieved evidence even though it fails the strict expected-section rubric.

Together, the two answer evaluations show both sides:

| Evaluation | Criterion | Result |
|---|---|---:|
| Strict checklist | Required facts + exact expected citations | **90.0%** |
| LLM judge | Groundedness / contradiction check | **100.0%** |

---

## Streamlit app

The web app is implemented in `app.py`.

Features:

- Modern Streamlit interface
- User query box
- Top-k retrieval slider
- Generated legal answer
- Retrieved PPC evidence shown below the answer
- FAISS score shown for each retrieved section
- Environment-configurable LLM model

The deployed app uses Groq because Streamlit Cloud cannot access local Ollama:

```python
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
```

Required Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_key_here"
LLM_MODEL = "llama-3.1-8b-instant"
```

The app also includes a small query expansion layer for common terminology-gap queries such as "murder", so that natural English queries are more likely to retrieve `qatl-e-amd` / Section 302 evidence.

---

## What is working

- **636 PPC sections indexed**
- **Section-level retrieval**
- **BGE-large embeddings**
- **Urdu/Arabic terminology normalisation**
- **Original legal text preserved for evidence**
- **FAISS vector store committed for deployment**
- **Streamlit app deployed with retrieved evidence display**
- **Groq-hosted LLM used for cloud answer generation**
- **Retrieval evaluation implemented**
- **Strict answer evaluation implemented**
- **LLM-as-judge groundedness evaluation implemented**
- **Retrieval performance:** 94.1% Hit@3, MRR 0.794
- **Strict answer evaluation:** 90.0% on selected 10-case set
- **LLM judge groundedness:** 100.0% on selected 10-case set

---

## What is not working

### 1. "Public servant taking bribe" misses Section 161

BGE-large ranks related provisions such as Sections 162, 165, and 171B ahead of Section 161. This is the known strict-evaluation failure.

### 2. Very short English queries can still be weak

Queries such as "murder punishment" may need terminology expansion to reliably retrieve `qatl-e-amd` and Section 302. The Streamlit app includes query expansion for common terms, but the underlying limitation remains.

### 3. Urdu script queries are not supported

The embedding model is English-focused. Urdu script input is not currently supported.

---

## Known limitations

1. **Dictionary-based normalisation**: new Urdu/Arabic terms require manual entries in `term_map`.
2. **Static index**: adding new laws requires rebuilding the FAISS vector store.
3. **Single statute**: only the Pakistan Penal Code is indexed.
4. **Retrieval-bound answers**: if retrieval misses the best section, the LLM can only answer from the retrieved evidence.
5. **Cloud model difference**: Streamlit Cloud uses Groq instead of local Ollama because local models are unavailable in hosted Streamlit.
6. **LLM judge leniency**: LLM-as-judge evaluation checks groundedness and contradictions, but is less strict than manually defined fact/citation checklists.

