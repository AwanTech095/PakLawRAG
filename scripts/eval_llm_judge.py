"""
LLM-as-judge evaluation for PakLawRAG.

This evaluates answer groundedness against the retrieved PPC evidence. It does
not compare the answer to a gold answer. The judge is instructed to decide
whether the generated answer is supported by, contradicted by, or unsupported by
the retrieved evidence.

Recommended setup:
    ollama pull llama3.2:latest

Run:
    python scripts/eval_llm_judge.py

Optional environment variables:
    ANSWER_MODEL=gemma3:4b
    JUDGE_MODEL=llama3.2:latest
    EVAL_LIMIT=10
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from eval_answers import TEST_CASES

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
_STORE_PATH = str(_ROOT / "vectorstore_sections")
_REPORT_PATH = _ROOT / "output" / "llm_judge_eval_report.json"

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemma3:4b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama3.2:latest")
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0") or "0")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
MAX_SECTION_CHARS = int(os.getenv("MAX_SECTION_CHARS", "900"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "140"))
MAX_JUDGE_TOKENS = int(os.getenv("MAX_JUDGE_TOKENS", "260"))

_vectorstore = None

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant specializing in Pakistani law.
Answer the user's question using ONLY the PPC sections provided below.
Be concise and cite the section numbers in your answer.
If the sections do not contain enough information to answer, say so.

Relevant PPC Sections:
{context}"""),
    ("human", "{question}"),
])

JUDGE_INSTRUCTIONS = """You are a strict legal RAG evaluator.
You must judge the generated answer ONLY against the retrieved PPC evidence.
Do not use outside legal knowledge.

Mark PASS only if:
1. The answer directly answers the question.
2. The important legal claims are supported by the evidence.
3. The answer does not contradict the evidence.
4. The answer uses section citations correctly.

Mark FAIL if:
- any major legal claim is contradicted by the evidence;
- the answer relies on a section not present in evidence for a central claim;
- the answer gives the wrong offence, wrong punishment, or wrong definition;
- the answer omits the central answer when the evidence contains it.

Return JSON only, with this exact shape:
{
  "verdict": "PASS" or "FAIL",
  "contradiction_found": true or false,
  "unsupported_major_claim": true or false,
  "missing_key_fact": true or false,
  "citation_correct": true or false,
  "reason": "one short explanation",
  "problem_claims": ["short claim 1", "short claim 2"]
}"""

JUDGE_USER_TEMPLATE = """Question:
{question}

Expected section IDs, if known:
{expected_sections}

Retrieved evidence:
{evidence}

Generated answer:
{answer}

Judge the generated answer now."""


@dataclass
class JudgeResult:
    id: str
    question: str
    expected_sections: list[str]
    retrieved_sections: list[str]
    answer_model: str
    judge_model: str
    generated_answer: str
    judge_verdict: str
    contradiction_found: bool
    unsupported_major_claim: bool
    missing_key_fact: bool
    citation_correct: bool
    passed: bool
    judge_reason: str
    problem_claims: list[str]
    raw_judge_output: str


def installed_ollama_models() -> set[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    models = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.add(parts[0])
    return models


def require_models() -> None:
    installed = installed_ollama_models()
    missing = [model for model in [ANSWER_MODEL, JUDGE_MODEL] if model not in installed]
    if missing:
        print("Missing Ollama model(s): " + ", ".join(missing))
        for model in missing:
            print(f"Pull with: ollama pull {model}")
        raise SystemExit(1)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            _STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(question: str):
    return get_vectorstore().similarity_search(question, k=RETRIEVAL_K)


def format_evidence(docs) -> str:
    blocks = []
    for doc in docs:
        section_id = doc.metadata.get("section_id", "")
        text = doc.metadata.get("original_text") or doc.page_content
        blocks.append(f"[Section {section_id}]\n{text[:MAX_SECTION_CHARS]}")
    return "\n\n".join(blocks)


def generate_answer(question: str, evidence: str) -> str:
    llm = ChatOllama(
        model=ANSWER_MODEL,
        temperature=0,
        num_predict=MAX_ANSWER_TOKENS,
    )
    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"question": question, "context": evidence})
    return response.content.strip()


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_answer(case: dict[str, Any], evidence: str, answer: str) -> tuple[dict[str, Any], str]:
    prompt = (
        JUDGE_INSTRUCTIONS
        + "\n\n"
        + JUDGE_USER_TEMPLATE.format(
            question=case["question"],
            expected_sections=", ".join(case.get("expected_sections", [])),
            evidence=evidence,
            answer=answer,
        )
    )
    judge = ChatOllama(
        model=JUDGE_MODEL,
        temperature=0,
        format="json",
        num_predict=MAX_JUDGE_TOKENS,
    )
    response = judge.invoke(prompt)
    raw = response.content.strip()
    return extract_json(raw), raw


def bool_field(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def run_case(case: dict[str, Any]) -> JudgeResult:
    docs = retrieve(case["question"])
    retrieved_sections = [doc.metadata.get("section_id", "") for doc in docs]
    evidence = format_evidence(docs)
    answer = generate_answer(case["question"], evidence)
    judge_payload, raw_judge = judge_answer(case, evidence, answer)

    verdict = str(judge_payload.get("verdict", "FAIL")).upper()
    contradiction_found = bool_field(judge_payload, "contradiction_found")
    unsupported_major_claim = bool_field(judge_payload, "unsupported_major_claim")
    missing_key_fact = bool_field(judge_payload, "missing_key_fact")
    citation_correct = bool_field(judge_payload, "citation_correct")

    passed = (
        verdict == "PASS"
        and not contradiction_found
        and not unsupported_major_claim
        and not missing_key_fact
        and citation_correct
    )

    problem_claims = judge_payload.get("problem_claims", [])
    if not isinstance(problem_claims, list):
        problem_claims = [str(problem_claims)]

    return JudgeResult(
        id=case["id"],
        question=case["question"],
        expected_sections=case.get("expected_sections", []),
        retrieved_sections=retrieved_sections,
        answer_model=ANSWER_MODEL,
        judge_model=JUDGE_MODEL,
        generated_answer=answer,
        judge_verdict=verdict,
        contradiction_found=contradiction_found,
        unsupported_major_claim=unsupported_major_claim,
        missing_key_fact=missing_key_fact,
        citation_correct=citation_correct,
        passed=passed,
        judge_reason=str(judge_payload.get("reason", "")),
        problem_claims=[str(claim) for claim in problem_claims],
        raw_judge_output=raw_judge,
    )


def run_eval() -> list[JudgeResult]:
    require_models()
    cases = TEST_CASES[:EVAL_LIMIT] if EVAL_LIMIT else TEST_CASES
    results = []

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['id']}: {case['question']}", flush=True)
        result = run_case(case)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  {status} | retrieved={','.join(result.retrieved_sections)} "
            f"| judge={result.judge_verdict} | reason={result.judge_reason}",
            flush=True,
        )

    return results


def save_report(results: list[JudgeResult]) -> None:
    passed = sum(result.passed for result in results)
    payload = {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "faithfulness_accuracy": passed / len(results) if results else 0,
            "answer_model": ANSWER_MODEL,
            "judge_model": JUDGE_MODEL,
        },
        "results": [asdict(result) for result in results],
    }

    _REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {_REPORT_PATH}")


if __name__ == "__main__":
    eval_results = run_eval()
    save_report(eval_results)
    total = len(eval_results)
    passed = sum(result.passed for result in eval_results)
    print(f"\nLLM judge faithfulness accuracy: {passed}/{total} ({passed / total:.1%})")
