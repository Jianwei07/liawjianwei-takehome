# Part B — LLM Evaluation Harness

A CLI tool that runs structured test cases against an LLM endpoint, scores each response with multiple complementary mechanisms, and produces a structured summary of pass rate, failures, and anomalies.

---

## What it does

1. **Reads** a JSONL test file (`{"id": "...", "input": "...", "expected": "..."}`)
2. **Calls** a configurable endpoint — mock (no network) or a real HTTP LLM API
3. **Scores** each response with three mechanisms (see Scoring below)
4. **Outputs** a summary: pass rate, per-failure reasons, anomaly flags
5. **Handles** malformed input and endpoint errors gracefully — bad lines are reported with line numbers; endpoint failures are recorded per-test without crashing the run

---

## Quickstart

```bash
cd part-b

# Create venv and install (creates `eval-harness` command)
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Run against the sample file using the default mock endpoint
eval-harness sample_tests.jsonl

# Verbose: print PASS/FAIL/ERROR per test as they run
eval-harness sample_tests.jsonl --verbose

# JSON output (machine-readable, useful for CI)
eval-harness sample_tests.jsonl --output json

# Different mock modes
eval-harness sample_tests.jsonl --endpoint mock:random
eval-harness sample_tests.jsonl --endpoint mock:fail   # all tests → ERROR

# Real HTTP endpoint
eval-harness sample_tests.jsonl --endpoint http://localhost:8080/generate

# Run unit tests
uv run pytest tests/ -v
```

### Without installing (run as a module)

```bash
cd part-b
uv run python -m harness.cli sample_tests.jsonl --verbose
```

---

## Scoring

Three mechanisms are applied to every response. A test **passes** if **any one** meets its threshold.

| Mechanism | How it works | Threshold | Why |
|---|---|---|---|
| **Exact match** | Normalised string equality (lowercase, strip punctuation, collapse whitespace) | 1.0 | Catches precisely correct factual answers (e.g. "14 days annual leave") |
| **Keyword F1** | Precision/recall/F1 over non-stopword token sets | ≥ 0.6 | Handles natural paraphrases where the key facts are present but phrasing differs |
| **Sequence similarity** | `difflib.SequenceMatcher` ratio on normalised strings | ≥ 0.7 | Catches near-correct responses with minor word differences |

**Rationale for "any one passes" logic:** LLM responses are rarely verbatim. A response that contains all the right keywords but adds a polite prefix ("Certainly! The answer is…") should still pass. Using the union of mechanisms reduces false negatives without requiring a heavy semantic model.

**Anomaly detection** flags:
- Empty responses
- All responses identical across different inputs (stuck endpoint)
- Responses taking > 10 seconds

---

## Endpoint modes

| Flag | Behaviour |
|---|---|
| `mock` (default) | Returns a fixed stub string |
| `mock:echo` | Echoes the input prompt (triggers anomaly: identical responses) |
| `mock:random` | Returns random HR-domain keywords |
| `mock:fail` | Raises an endpoint error for every call |
| `mock:timeout` | Simulates a timeout error |
| `http://...` | Calls a real endpoint (POST, JSON body `{"prompt": "..."}`) |

---

## Project layout

```
part-b/
├── harness/
│   ├── cli.py        — argparse CLI, argument parsing, output formatting
│   ├── runner.py     — JSONL loading, test loop, anomaly detection
│   ├── scorer.py     — exact_match, keyword_f1, sequence_similarity
│   └── endpoint.py   — MockEndpoint, HttpEndpoint, EndpointError
├── tests/
│   └── test_scorer.py — unit tests (normalize, exact_match, keyword_f1,
│                          sequence_similarity, score, load_test_cases)
├── sample_tests.jsonl — 5 HR-policy test cases
├── pyproject.toml    — package config + entry point
└── requirements.txt  — pytest only; runtime has zero third-party deps
```

---

## Assumptions

- Python 3.11+ (uses `list[...]` / `set[...]` type hints without `from __future__ import annotations`)
- The real HTTP endpoint accepts `POST {"prompt": "..."}` and returns one of: `{"response": "..."}`, `{"choices": [{"message": {"content": "..."}}]}`, or `{"text": "..."}`
- "Snappy" scoring is intentionally done without heavy ML dependencies (no `sentence-transformers`, `nltk`, etc.) to keep the harness portable and fast
- Keyword F1 threshold of 0.6 reflects a pragmatic choice: too low (0.4) accepts nonsense answers; too high (0.8) penalises valid paraphrases
- Test IDs in the JSONL file are treated as opaque strings — no uniqueness enforcement

---

## What I'd add with more time

1. **Semantic similarity scoring** using a local embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`) — cosine similarity > 0.75 as a fourth mechanism for capturing meaning without exact keyword overlap
2. **Async concurrency** — run multiple test cases in parallel against the endpoint for faster evaluation of large suites
3. **Configurable thresholds** — pass thresholds as CLI flags or a YAML config file so teams can tune them per-project
4. **HTML report** — richer output with per-test score breakdowns, sortable tables, trend charts across multiple runs
5. **Retry with backoff** — automatic retry on transient endpoint errors (5xx, timeout) with exponential backoff
6. **Streaming support** — handle Server-Sent Events / streaming responses from vLLM-style endpoints
