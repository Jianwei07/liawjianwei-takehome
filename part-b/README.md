# Part B - Grounded RAG Evaluation Harness

A lightweight CLI and local HTTP service for evaluating grounded answers from an LLM/RAG-style endpoint.

This Part B implementation is designed to complement Part A.

- Part A defines the retrieval and answer pipeline: `BM25 + FAISS -> RRF -> reranker -> grounded answer`.
- Part B acts as the acceptance layer for that design: it checks answer quality, source grounding, abstention behavior, and failure buckets that point to likely next actions.

The project stays intentionally small:

- stdlib-only scoring and HTTP client/server
- JSONL knowledge base instead of a full retrieval stack
- JSONL eval suites instead of a database or dashboard
- structured JSON artifacts that can be saved after each run

Bundled inputs live in `data/`. Generated eval outputs can be written to `results/`, which is gitignored.

---

## What This Simulates

Part A describes a grounded internal document Q&A system. Part B does not rebuild the full production retrieval stack. Instead, it simulates the outer contract that the eval layer cares about:

```text
user query -> endpoint -> answer + cited sources -> eval summary
```

`data/knowledge_base.jsonl` stands in for indexed internal documents. `KnowledgeBaseEndpoint` performs simple keyword retrieval over those records and returns:

- an `answer`
- a list of cited `sources`

If retrieval confidence is too weak, the endpoint returns a canonical abstention:

```json
{
  "answer": "I don't know based on the provided documents.",
  "sources": []
}
```

That lets the harness measure whether the system abstains cleanly instead of hallucinating.

---

## Pipeline Flow

```text
pyproject.toml
├─ uv sync --extra dev
├─ uv run pytest tests/ -v       -> automated verification
├─ uv run task dev               -> start backend on port 8080
├─ curl -X POST /eval            -> run backend eval
├─ uv run eval-harness ...       -> CLI eval alternative
└─ uv run eval-endpoint ...      -> backend with custom options

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
│ kb:... -> KnowledgeBaseEndpoint                                      │
│ http://... -> HttpEndpoint                                           │
│ mock:* -> diagnostic endpoint behavior                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ harness/scorer.py                                                    │
│ exact_match + answer_coverage + keyword_f1 + sequence_similarity     │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ stdout / sample_results*.json                                        │
│ pass rate, grounding rate, failure buckets, anomalies, per-test rows │
└──────────────────────────────────────────────────────────────────────┘

CLI EVAL ARTIFACT FLOW
┌──────────────────────────────────────────────────────────────────────┐
│ uv run eval-harness data/tests_positive.jsonl --endpoint kb:...       │
│   --save results/positive.json                                        │
│ Same runner/scorer path, without starting the HTTP server            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Why This Complements Part A

Part A is concerned with retrieval quality, grounding, and monthly KB refreshes.

That means the most useful eval questions are not just:

- did the answer text roughly match

They are also:

- did the answer come from an expected source
- did the system abstain when the KB should not answer
- if it failed, was it an answer problem, a grounding problem, or a runtime problem

This harness keeps those signals small and actionable without introducing heavy eval infrastructure.

---

## Positive And Diagnostic Suites

The repo ships two suites.

- `data/tests_positive.jsonl`: positive release-gating suite
- `data/tests_diagnostic.jsonl`: diagnostic suite for robustness and grounding checks

The positive suite is the acceptance gate.

The diagnostic suite is not meant to be a pure pass/fail release blocker. It is meant to surface useful failure modes such as:

- unknown queries that should abstain
- correct answers grounded in the wrong source document
- current-vs-archived policy confusion

This mirrors the real Part A risk profile more closely than a generic “wrong expected answer” test set.

---

## Test Schema

The harness accepts the original take-home shape:

```json
{
  "id": "q1",
  "input": "What is the annual leave policy?",
  "expected": "14 days annual leave"
}
```

It also supports a slightly richer typed schema used by the bundled suites:

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
- `tags`: optional labels for later filtering or inspection

---

## Endpoint Contract

`POST /generate` accepts:

```json
{ "prompt": "What is the annual leave policy?" }
```

It returns:

```json
{
  "answer": "Employees receive 14 days annual leave per calendar year.",
  "sources": ["HR/2026/leave-policy.md"]
}
```

`POST /eval` runs the configured suite through the same endpoint behavior and returns the structured eval summary.

Optional diagnostic suite override:

```json
{ "tests": "tests_diagnostic.jsonl" }
```

Routes:

- `GET /health`
- `POST /generate`
- `POST /eval`

---

## Scoring And Verdicts

Answer text scoring remains intentionally lightweight and fully reproducible.

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

- the answer text passes one of the lexical thresholds
- and at least one expected source matches when `expected_sources` are provided

An `abstain` case passes when the endpoint returns the canonical abstention answer.

---

## Failure Buckets

The summary classifies misses into a small actionable set:

- `answer_mismatch`: text did not meet the answer thresholds
- `source_mismatch`: answer text was acceptable, but the cited source was not an expected one
- `abstain_miss`: the system should have abstained but answered anyway
- `endpoint_error`: runtime or transport failure

These buckets help separate likely next steps:

- `answer_mismatch` with correct source suggests prompt/generation issues
- `source_mismatch` suggests retrieval or reranking issues
- `abstain_miss` suggests retrieval-threshold or grounding-policy issues
- `endpoint_error` suggests service/runtime problems

---

## Summary Output

The JSON summary includes:

- total tests
- passed / failed / endpoint errors
- pass rate
- grounding rate
- abstention success rate
- counts by failure bucket
- anomalies
- per-test answer, sources, expected sources, scores, latency, and reason

This makes the saved artifact useful for:

- release/UAT decisions on the positive suite
- diagnostic inspection on the negative suite
- comparing eval artifacts after monthly knowledge-base refreshes

---

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

In another terminal, call the local endpoint:

```bash
curl -s -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the annual leave policy?"}'
```

Run the positive suite through the backend:

```bash
curl -s -X POST http://127.0.0.1:8080/eval
```

Run the diagnostic suite through the backend:

```bash
curl -s -X POST http://127.0.0.1:8080/eval \
  -H "Content-Type: application/json" \
  -d '{"tests":"tests_diagnostic.jsonl"}'
```

The `tests` override is restricted to the configured test directory, so the backend only accepts JSONL files from `part-b/data/` by default.

Generate the same positive artifact from the CLI:

```bash
mkdir -p results

uv run eval-harness data/tests_positive.jsonl \
  --endpoint kb:data/knowledge_base.jsonl \
  --output json \
  --save results/positive.json
```

Generate the diagnostic artifact from the CLI:

```bash
uv run eval-harness data/tests_diagnostic.jsonl \
  --endpoint kb:data/knowledge_base.jsonl \
  --output json \
  --save results/diagnostic.json
```

No manual virtualenv activation is needed. `uv` reads `pyproject.toml`, installs the package, exposes the `eval-harness`, `eval-endpoint`, and `task` entrypoints, and runs commands inside the project environment.

---

## Endpoint Modes

- `kb:data/knowledge_base.jsonl`: local simulated grounded endpoint for this submission
- `mock` or `mock:fixed`: always returns the same response
- `mock:echo`: echoes the input query
- `mock:random`: returns random HR-like keywords
- `mock:fail`: simulates endpoint failure handling
- `mock:timeout`: simulates timeout handling
- `http://...` or `https://...`: calls a real endpoint, including the local `main.py` service

---

## Data Directory

`data/` contains the small bundled dataset used by this take-home:

- `knowledge_base.jsonl`: simulated internal documents
- `tests_positive.jsonl`: release-gating checks
- `tests_diagnostic.jsonl`: grounding and abstention probes

`results/` is intentionally not committed. Use it for local eval artifacts generated from CLI or `/eval` runs.

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
├── results/
│   └── .gitkeep
├── harness/
│   ├── __init__.py
│   ├── cli.py
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
- store multiple historical artifacts and diff them across monthly KB rebuilds
- expand the suites by department, query type, and time-sensitive policy changes
- add semantic similarity as a secondary scorer for harder paraphrases
