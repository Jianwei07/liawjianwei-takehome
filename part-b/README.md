# Part B — LLM Evaluation Harness

A CLI tool that evaluates whether an LLM endpoint answers correctly, built to simulate evaluating the Part A Q&A system.

`sample_results.json` — pre-generated output. Open it to see what the harness produces without running anything.

---

## What this simulates

Part A describes a RAG pipeline: HR-policy documents → BM25 + FAISS retrieval → RRF fusion → cross-encoder reranker → LLM answer via vLLM.

This harness simulates that pipeline end-to-end using a local JSONL knowledge base. It sends queries through retrieval, receives grounded answers, and scores them against ground-truth expectations.

```
sample_knowledge_base.jsonl         sample_tests.jsonl
  │                                    │
  │  {"id": "kb_leave_policy",         │  {"id": "q1",
  │   "source": "HR/leave-policy.md",  │   "input": "What is the annual leave policy?",
  │   "text": "Full-time employees...",│   "expected": "14 days annual leave"}
  │   "answer": "Employees receive..."}│
  ▼                                    ▼
KnowledgeBaseEndpoint              runner.load_test_cases()
  - keyword retrieval over KB          - validates JSONL, collects all errors upfront
  - returns answer + source citation   │
  │                                    │
  └────────────────────────────────────┘
                    │
                    ▼
            scorer.score(response, expected)
              - exact_match, answer_coverage, keyword_f1, sequence_similarity
                    │
                    ▼
              RunSummary
                - pass rate, per-failure reasons, anomalies
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
scorer.py    math only      — four scoring functions, pure stdlib, no I/O
endpoint.py  I/O only       — KB simulator, mock stub, real HTTP; unified error type
```

**Why this separation matters:**

`scorer.py` has no I/O and no argparse dependency — it can be unit-tested directly with no mocking.

`runner.py` knows nothing about HTTP or CLI arguments. Swapping `KnowledgeBaseEndpoint` for `HttpEndpoint` (the real Part A service) requires zero changes to the runner.

`cli.py` contains no business logic. Changing output format (text vs JSON) requires zero changes to scorer or runner.

---

## Why KnowledgeBaseEndpoint instead of a pure mock

A `MockEndpoint` returning fixed or random strings tests the scoring machinery, but tells the interviewer nothing about whether the harness can evaluate a real retrieval pipeline.

`KnowledgeBaseEndpoint` simulates the Part A pipeline at the right level of abstraction:

- `sample_knowledge_base.jsonl` stands in for the BM25 + FAISS indexed documents.
- The endpoint retrieves the best-matching record using keyword overlap (analogous to BM25 scoring).
- It returns the answer with a source citation, matching what a real RAG endpoint would return: `"Employees receive 14 days annual leave per calendar year. Source: HR/2026/leave-policy.md."`

This makes `sample_results.json` meaningful — it shows actual retrieval + grounded answers, not stubs.

---

## Scoring — why four mechanisms and how each works

LLM responses are never verbatim. "14 days annual leave" and "Employees receive 14 days annual leave per calendar year. Source: HR/2026/leave-policy.md." are both correct answers. A single scoring mechanism would reject one of them.

A test **passes** if **any one** mechanism meets its threshold. OR logic reduces false negatives.

### 1. `exact_match` — threshold 1.0

Normalise both strings (lowercase, strip punctuation, collapse whitespace), then compare equality. Binary: 1.0 or 0.0.

Catches verbatim correct answers. Too strict for natural LLM output — but costs nothing and catches the easy case first.

### 2. `answer_coverage` — threshold 1.0

The primary mechanism for RAG evaluation.

```
exp_kw  = expected keywords (stopwords removed)
res_kw  = response keywords (stopwords removed)
coverage = |res_kw ∩ exp_kw| / |exp_kw|   ← recall only, no precision penalty
```

Why recall-only and not F1: RAG responses naturally contain more than the expected answer. A response of "Employees receive 14 days annual leave per calendar year. Source: HR/2026/leave-policy.md." is correct even though it contains far more words than "14 days annual leave". Precision would penalise those extra words. Coverage only checks: did the response contain all the key facts from the expected answer?

Threshold 1.0: all expected keywords must be present. Because expected answers in the test set are short and factual, a coverage of 1.0 is the right bar — partial keyword coverage means a key fact is missing.

### 3. `keyword_f1` — threshold 0.6

Balances precision and recall on keyword sets. Catches responses that miss some expected terms but aren't fully wrong:

```
precision = overlap / response_keywords   ← penalises hallucinated terms
recall    = overlap / expected_keywords   ← penalises missing terms
F1        = 2 × precision × recall / (precision + recall)
```

Threshold 0.6: below 0.6, too many key terms are missing or replaced. This backstop handles cases where coverage is < 1.0 but the answer is still substantially correct.

### 4. `sequence_similarity` — threshold 0.7

`difflib.SequenceMatcher.ratio()` = `2 × matching_characters / total_characters`. Catches near-identical answers with minor word differences — e.g. "within 30 days of the expense" vs "within 30 days of incurring the expense".

### Anomaly detection

Beyond per-test scoring, the runner flags three system-level signals:

- **Empty responses** — vLLM returns empty string on OOM or context overflow. Without this flag, it shows as a normal FAIL, hiding the real cause.
- **All responses identical** — signals a stuck endpoint or caching bug. Fires on `mock:fixed` by design.
- **Latency > 10s** — signals GPU pressure or a queued request in the Part A serving stack.

---

## Error handling

**Malformed JSONL:** `load_test_cases()` collects all parse errors before raising. If lines 2 and 47 are both broken, you see both in one run.

**Endpoint failures:** `EndpointError` is the single exception type all endpoints raise. `runner.py` catches only `EndpointError` — it records the failure per test and continues. One bad request never crashes the run. Exit code 1 signals CI that errors occurred.

Why a unified exception: `HttpEndpoint` can throw `urllib.error.HTTPError`, `urllib.error.URLError`, or `TimeoutError`. If `runner.py` caught all three, adding a new endpoint type would require changing the runner. `EndpointError` is the contract — runner never needs to know what it's talking to.

---

## Quickstart

```bash
cd part-b
uv venv && uv pip install -e ".[dev]"
source .venv/bin/activate

# Run against the simulated KB (recommended — shows full pipeline)
eval-harness sample_tests.jsonl --endpoint kb:sample_knowledge_base.jsonl --verbose

# Save full JSON results to file
eval-harness sample_tests.jsonl --endpoint kb:sample_knowledge_base.jsonl --save results.json

# Run against real Part A endpoint
eval-harness sample_tests.jsonl --endpoint http://localhost:8080/generate --verbose

# Run endpoint failure demo
eval-harness sample_tests.jsonl --endpoint mock:fail --output json

# Unit tests
uv run pytest tests/ -v
```

---

## Project layout

```
part-b/
├── sample_knowledge_base.jsonl  — 5 HR-policy KB records (simulates Part A indexed docs)
├── sample_tests.jsonl           — 5 HR-policy test cases aligned with the KB
├── sample_results.json          — pre-generated output (read without running)
├── harness/
│   ├── cli.py                   — user interface only, no business logic
│   ├── runner.py                — orchestration: load, call, score, anomaly detect
│   ├── scorer.py                — four scoring functions, pure stdlib
│   └── endpoint.py              — KnowledgeBaseEndpoint + MockEndpoint + HttpEndpoint
└── tests/
    └── test_scorer.py           — unit tests covering scoring, loading, and pipeline
```

---

## What I would add with more time

**RAGAS integration** — RAGAS provides RAG-specific metrics (faithfulness, context precision/recall) that require access to the retrieved chunks, not just the final answer. The current harness is a black-box scorer; RAGAS is a white-box RAG evaluator. Both are needed: this harness catches regressions, RAGAS diagnoses which layer broke.

**Semantic similarity** — cosine similarity via `all-MiniLM-L6-v2` (local, no cloud). Catches paraphrase equivalents that lexical methods miss — e.g. "two weeks annual leave" vs "14 days annual leave" scores zero on keyword_f1 but high on embeddings. Zero cloud dependency, compatible with air-gapped environments.

**Async concurrency** — run test cases in parallel against the endpoint. Relevant for large suites against a real vLLM endpoint where throughput matters.

**Configurable thresholds via YAML** — teams tune scoring thresholds per-project without touching code. Current values are starting points that should be calibrated against a labelled golden set.

**HTML report with trend tracking** — per-test score breakdown stored in local SQLite across runs. A drop in answer_coverage after a monthly rebuild — before users notice — is the Part C early warning signal.
