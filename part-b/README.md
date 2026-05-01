# Part B - Grounded RAG Evaluation Service

A small HTTP backend that runs JSONL evaluation suites against a grounded
answering endpoint and returns a structured JSON summary.

This stays intentionally small and submission-friendly:

- one backend service, no separate CLI workflow
- stdlib-only scoring and HTTP server/client
- JSONL fixtures instead of a full retrieval stack
- structured summaries for release-gating and diagnostics
- explicit grounding and abstention checks to complement Part A

## Quickstart

From the repo root:

```bash
cd part-b
uv sync --extra dev
```

Run the automated verification suite:

```bash
uv run pytest tests/ -v
```

Start the local backend:

```bash
uv run task dev
```

Open a second terminal anywhere inside the repo. Use an explicit `PART_B_DIR`
path so saved JSON always lands in `part-b/`.

```bash
PART_B_DIR="$(git rev-parse --show-toplevel)/part-b"
```

Check one grounded answer from `/generate`:

```bash
curl -sS -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the annual leave policy?"}'
```

Save the positive eval summary as a JSON artifact in `part-b/`:

```bash
curl -sS -X POST http://127.0.0.1:8080/eval \
  -o "$PART_B_DIR/positive.json" && ls -lh "$PART_B_DIR/positive.json"
```

Save the diagnostic eval summary as a JSON artifact in `part-b/`:

```bash
curl -sS -X POST http://127.0.0.1:8080/eval \
  -H "Content-Type: application/json" \
  -d '{"tests":"tests_diagnostic.jsonl"}' \
  -o "$PART_B_DIR/diagnostic.json" && ls -lh "$PART_B_DIR/diagnostic.json"
```

Watch the terminal running `uv run task dev` for progress logs like:

```text
[01/May/2026 15:12:01] eval start suite=tests_diagnostic.jsonl
[01/May/2026 15:12:01] eval done suite=tests_diagnostic.jsonl passed=3/6 failed=3 errors=0
```

Inspect the saved output:

```bash
cat "$PART_B_DIR/positive.json"
cat "$PART_B_DIR/diagnostic.json"
```

The `tests` override is restricted to the configured test directory, so the
backend only accepts JSONL files from `part-b/data/` by default.

Because the examples use `$(git rev-parse --show-toplevel)/part-b`, they work
the same from the repo root or from inside `part-b/`.

## Expected Output

Positive response from `/generate`:

```json
{
  "answer": "Employees receive 14 days annual leave per calendar year.",
  "sources": ["HR/2026/leave-policy.md"]
}
```

Neutral response from `/generate` when the KB should abstain:

```json
{
  "answer": "I don't know based on the provided documents.",
  "sources": []
}
```

Negative/diagnostic response from `/eval` is still a valid JSON summary, but it
contains expected failures:

```json
{
  "total": 6,
  "passed": 3,
  "failed": 3,
  "errors": 0,
  "failure_buckets": {
    "source_mismatch": 3
  }
}
```

`POST /eval` returns one structured JSON summary. `curl` prints it to stdout by
default; `curl -o <file>` persists it as an artifact, so a successful request
can look silent in the caller terminal while the JSON is written to disk.
The examples above save directly to `part-b/positive.json` and
`part-b/diagnostic.json`.

---

## What To Review

If I were reviewing this submission as an interviewer, I would read in this
order:

1. `part-b/README.md` for system shape and tradeoffs
2. `part-b/main.py` for the HTTP contract
3. `part-b/harness/runner.py` for verdict logic and failure buckets
4. `part-b/harness/endpoint.py` for the simulated grounded endpoint
5. `part-b/data/` for the bundled knowledge base and test suites
6. `part-b/tests/` for behavioral coverage

---

## Why This Exists

Part A defines the serving design:

```text
BM25 + FAISS -> RRF -> reranker -> grounded answer
```

Part B complements that design by acting as the acceptance layer.

The main questions it answers are:

- was the answer correct enough for release
- was it grounded in an expected source
- did it abstain when the KB should not answer
- if it failed, was it an answer, grounding, or runtime problem

That keeps Part B aligned with the real failure surface of Part A rather than
turning it into a generic text-matching harness.

---

## Pipeline Flow

```text
pyproject.toml
├─ uv sync --extra dev
├─ uv run pytest tests/ -v       -> automated verification
├─ uv run task dev               -> start backend on port 8080
├─ curl -X POST /generate        -> one grounded answer
├─ curl -X POST /eval            -> eval summary JSON
└─ curl -o *.json                -> saved local artifacts

AUTOMATED VERIFICATION
┌──────────────────────────────────────────────────────────────────────┐
│ tests/                                                               │
│ - test_scorer.py: scoring, JSONL loading, runner behavior            │
│ - test_backend.py: /health, /generate, /eval backend routes          │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ harness modules + main.py                                            │
│ pytest checks the scorer, runner, KB endpoint, and HTTP wrapper      │
└───────────────────────────────┬──────────────────────────────────────┘

HTTP EVAL FLOW
┌───────────────────────┐       ┌──────────────────────────────────────┐
│ data/tests_*.jsonl    │       │ data/knowledge_base.jsonl            │
│ queries + expected    │       │ simulated internal knowledge base    │
└───────────┬───────────┘       └──────────────────┬───────────────────┘
            │                                      │
            ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│ uv run task dev -> main.py                                           │
│ POST /generate -> KnowledgeBaseEndpoint -> answer + sources          │
│ POST /eval     -> runner.run(data/tests_*.jsonl, KB) -> JSON         │
│ optional body for diagnostic suite:                                  │
│   {"tests":"tests_diagnostic.jsonl"}                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ harness/runner.py                                                    │
│ load tests -> call endpoint per query -> score -> bucket failures    │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ harness/endpoint.py                                                  │
│ KnowledgeBaseEndpoint -> grounded answer or abstain                  │
│ HttpEndpoint          -> real upstream endpoint adapter              │
│ MockEndpoint          -> diagnostic endpoint behavior                │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ harness/scorer.py                                                    │
│ exact_match + answer_coverage + keyword_f1 + sequence_similarity     │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ stdout / positive.json diagnostic.json                               │
│ pass rate, grounding rate, failure buckets, anomalies, per-test rows │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Strategy

The service uses two bundled suites:

- `data/tests_positive.jsonl`: release-gating checks
- `data/tests_diagnostic.jsonl`: robustness and grounding probes

The positive suite is the gate.

The diagnostic suite exists to surface useful failure modes such as:

- unknown queries that should abstain
- correct answer text grounded in the wrong document
- current-vs-archived policy confusion

This is a better fit for Part A than a purely black-box answer-only test set.

### Backend Routes

- `GET /health`
- `POST /generate`
- `POST /eval`

`POST /eval` accepts the default positive suite, or a diagnostic override:

```json
{ "tests": "tests_diagnostic.jsonl" }
```

### Verdict Model

Answer text scoring remains intentionally lightweight and reproducible.

Metrics:

- `exact_match`
- `answer_coverage`
- `keyword_f1`
- `sequence_similarity`

Thresholds:

- `exact_match >= 1.0`
- `answer_coverage >= 1.0`
- `keyword_f1 >= 0.6`
- `sequence_similarity >= 0.7`

An `answer` case passes when:

- the answer text clears one lexical threshold
- and at least one expected source matches when source expectations exist

An `abstain` case passes when the endpoint returns the canonical abstention
answer.

### Failure Buckets

Misses are grouped into a small actionable set:

- `answer_mismatch`
- `source_mismatch`
- `abstain_miss`
- `endpoint_error`

These buckets are meant to suggest the next investigation step:

- `answer_mismatch` with correct source suggests prompt/generation issues
- `source_mismatch` suggests retrieval or reranking issues
- `abstain_miss` suggests retrieval-threshold or grounding-policy issues
- `endpoint_error` suggests service/runtime problems

### Summary Output

The JSON summary includes:

- total tests
- passed / failed / endpoint errors
- pass rate
- grounding rate
- abstention success rate
- counts by failure bucket
- anomalies
- per-test answer, sources, expected sources, scores, latency, and reason

---

## Data Layout

`data/` contains the bundled inputs for this take-home:

- `knowledge_base.jsonl`: simulated internal documents
- `tests_positive.jsonl`: release-gating checks
- `tests_diagnostic.jsonl`: grounding and abstention probes

The harness still accepts the original take-home JSONL shape:

```json
{
  "id": "q1",
  "input": "What is the annual leave policy?",
  "expected": "14 days annual leave"
}
```

The bundled suites use a slightly richer typed schema:

```json
{
  "id": "p1_leave_policy",
  "input": "What is the annual leave policy?",
  "case_type": "answer",
  "expected_answer": "14 days annual leave",
  "expected_sources": ["HR/2026/leave-policy.md"],
  "tags": ["positive", "policy"]
}
```

Abstain case:

```json
{
  "id": "d1_unknown_dental",
  "input": "What is the dental benefits allowance?",
  "case_type": "abstain",
  "tags": ["diagnostic", "unknown"]
}
```

Fields:

- `id`: test case identifier
- `input`: query sent to the endpoint
- `case_type`: `answer` or `abstain` (defaults to `answer`)
- `expected_answer`: expected answer text for `answer` cases
- `expected_sources`: acceptable cited source paths
- `tags`: optional labels for filtering or inspection

`positive.json` and `diagnostic.json` are intentionally not committed. Generate
them locally from `/eval` responses.

---

## Project Layout

```text
part-b/
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── README.md
├── main.py
├── data/
│   ├── knowledge_base.jsonl
│   ├── tests_positive.jsonl
│   └── tests_diagnostic.jsonl
├── harness/
│   ├── __init__.py
│   ├── endpoint.py
│   ├── reporting.py
│   ├── runner.py
│   └── scorer.py
└── tests/
    ├── __init__.py
    ├── test_backend.py
    └── test_scorer.py
```

---

## With More Time

- add retrieval candidate debug fields when evaluating a real Part A endpoint
- store historical artifacts and diff them across monthly KB rebuilds
- expand the suites by department, query type, and time-sensitive changes
- add semantic similarity as a secondary scorer for harder paraphrases
