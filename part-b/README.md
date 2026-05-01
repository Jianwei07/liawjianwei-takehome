# Part B — LLM Evaluation Harness

A CLI tool that evaluates whether an LLM endpoint answers correctly, built to simulate evaluating the Part A Q&A system.

`sample_results.json` — pre-generated output. Open it to see what the harness produces without running anything.

---

## What this simulates

Part A describes a RAG pipeline: HR-policy documents → BM25 + FAISS retrieval → RRF fusion → cross-encoder reranker → LLM answer via vLLM.

This harness treats that pipeline as a black box. It sends queries to the endpoint, receives answers, and scores them against ground-truth expectations — the same HR-policy questions the Part A system is designed to answer.

```
sample_tests.jsonl
  │
  │  {"id": "q1", "input": "What is the annual leave policy?",
  │               "expected": "14 days annual leave"}
  ▼
runner.load_test_cases()           ← validates JSONL, collects all errors upfront
  │
  ▼
endpoint.call(input)               ← MockEndpoint (fixed / random / echo / fail)
  │                                   HttpEndpoint → Part A FastAPI service
  ▼
scorer.score(response, expected)   ← exact_match, keyword_f1, sequence_similarity
  │
  ▼
RunSummary                         ← pass rate, per-failure reasons, anomalies
  │
  ├── stdout (text or JSON)
  └── results.json (--save flag)
```

---

## Architecture — why four files

The harness is split so each module has exactly one responsibility:

```
cli.py       user interface — argparse, output formatting, exit codes
runner.py    orchestration  — load → call → score → collect → detect anomalies
scorer.py    math only      — three scoring functions, pure stdlib, no I/O
endpoint.py  I/O only       — mock stub and real HTTP, unified error type
```

**Why this separation matters:**

`scorer.py` has no I/O and no argparse dependency — it can be unit-tested directly with no mocking. All 33 unit tests target this file alone.

`runner.py` knows nothing about HTTP or CLI arguments. Swapping `MockEndpoint` for `HttpEndpoint` (the real Part A service) requires zero changes to the runner.

`cli.py` contains no business logic. Changing output format (text vs JSON) requires zero changes to scorer or runner.

`demo.py` drives the entire harness programmatically by importing `runner` and `endpoint` directly — it never touches `cli.py`. This proves the architecture: the CLI is just one consumer of the core logic, not the core itself.

---

## Scoring — why three mechanisms and how each works

LLM responses are never verbatim. "14 days annual leave" and "You are entitled to 14 days of annual leave per year" are both correct answers. A single scoring mechanism would reject one of them.

A test **passes** if **any one** mechanism meets its threshold. OR logic reduces false negatives without requiring a heavy ML dependency.

### 1. `exact_match` — threshold 1.0

Normalise both strings (lowercase, strip punctuation, collapse whitespace), then compare equality. Binary: 1.0 or 0.0.

Catches verbatim correct answers. Too strict for natural LLM output on its own — but costs nothing to compute and catches the easy case first.

### 2. `keyword_f1` — threshold 0.6

Tokenise both strings, remove stopwords, compute F1 on the resulting keyword sets:

```
precision = matched_keywords / response_keywords   ← penalises hallucinated terms
recall    = matched_keywords / expected_keywords   ← penalises missing terms
F1        = 2 × precision × recall / (precision + recall)
```

Why F1 and not just recall: recall alone rewards dumping every possible keyword ("the leave policy manager HR team 14 days 30 days reimbursement…"). Precision penalises that. F1 is the balance between covering required terms and not hallucinating extras.

Threshold 0.6: below 0.6, too many key terms are missing or replaced to consider the answer correct. Above 0.8 would reject valid paraphrases.

### 3. `sequence_similarity` — threshold 0.7

`difflib.SequenceMatcher.ratio()` = `2 × matching_characters / total_characters`. Finds longest common subsequences and scores structural similarity.

Catches near-identical answers with minor word differences that keyword_f1 would penalise — e.g. "within 30 days of the expense" vs "within 30 days of incurring the expense".

### Anomaly detection

Beyond per-test scoring, the runner flags three system-level signals:

- **Empty responses** — vLLM returns empty string on OOM or context overflow. Without this flag, it would show as a normal FAIL, hiding the real cause.
- **All responses identical** — every non-error response is the same string. Signals a stuck endpoint or caching bug. Fires on `mock:fixed` by design — confirms the detection works.
- **Latency > 10s** — signals GPU pressure or a queued request in the Part A serving stack.

---

## Error handling

**Malformed JSONL:** `load_test_cases()` collects all parse errors before raising. If lines 2 and 47 are both broken, you see both in one run — not one at a time.

**Endpoint failures:** `EndpointError` is the single exception type both `MockEndpoint` and `HttpEndpoint` raise. `runner.py` catches only `EndpointError` — it records the failure per test and continues. One bad request never crashes the run. Exit code 1 signals CI that errors occurred.

Why a unified exception: `HttpEndpoint` can throw `urllib.error.HTTPError`, `urllib.error.URLError`, or `TimeoutError`. If `runner.py` caught all three, adding a new endpoint type would require changing the runner. `EndpointError` is the contract — runner never needs to know what it's talking to.

---

## Quickstart

```bash
cd part-b
uv venv && uv pip install -e ".[dev]"
source .venv/bin/activate

# Run with mock endpoint (no network)
eval-harness sample_tests.jsonl --verbose

# Save full JSON results to file
eval-harness sample_tests.jsonl --save results.json

# Run against real Part A endpoint
eval-harness sample_tests.jsonl --endpoint http://localhost:8080/generate --verbose

# Interactive walkthrough (scoring demo, mock modes, Part C diagnostic)
uv run python demo.py

# Unit tests
uv run pytest tests/ -v
```

---

## Project layout

```
part-b/
├── sample_results.json    — pre-generated output (read without running)
├── sample_tests.jsonl     — 5 HR-policy test cases (aligned with Part A)
├── demo.py                — programmatic walkthrough, no CLI required
├── harness/
│   ├── cli.py             — user interface only, no business logic
│   ├── runner.py          — orchestration: load, call, score, anomaly detect
│   ├── scorer.py          — three scoring functions, pure stdlib
│   └── endpoint.py        — MockEndpoint (5 modes) + HttpEndpoint
└── tests/
    └── test_scorer.py     — 33 unit tests covering all scoring edge cases
```

---

## What I would add with more time

**RAGAS integration** — RAGAS provides RAG-specific metrics (faithfulness, context precision/recall) that require access to the retrieved chunks, not just the final answer. The current harness is a black-box scorer; RAGAS is a white-box RAG evaluator. Both are needed: this harness catches regressions, RAGAS diagnoses which layer broke.

**Semantic similarity as a fourth mechanism** — cosine similarity via `all-MiniLM-L6-v2` (local, no cloud). Captures meaning overlap without exact keyword matching. Zero cloud dependency, compatible with the air-gapped Part A environment.

**Async concurrency** — run test cases in parallel against the endpoint. Relevant for large suites against a real vLLM endpoint where throughput matters.

**Configurable thresholds via YAML** — teams tune scoring thresholds per-project without touching code. Current values (kf1 ≥ 0.6, ss ≥ 0.7) are starting points that should be calibrated against a labelled golden set.

**HTML report with trend tracking** — per-test score breakdown stored in local SQLite across runs. A drop in keyword_f1 after a monthly rebuild — before users notice — is the Part C early warning signal.
