# HTX AI Engineer Take-Home — Liau Jian Wei

Submission for the HTX LLM/LMM AI R&D Deployment Team take-home assignment.

---

## Repo structure

```
.
├── part-a/               Part A: System design (PDF source)
│   └── system-design.md
├── part-b/               Part B: LLM evaluation harness (runnable code)
│   ├── harness/          Core library (scorer, runner, endpoint, CLI)
│   ├── tests/            Unit tests (pytest)
│   ├── sample_tests.jsonl
│   ├── pyproject.toml
│   └── README.md         ← Start here for Part B
├── part-c/               Part C: Opinion piece (PDF source)
│   └── opinion.md
└── report-gen/           PDF generation tooling (md-to-pdf via headless Chrome)
```

---

## Part A — System Design

**File:** [part-a/system-design.md](part-a/system-design.md)  
**PDF:** generated via `report-gen` (see below)

Covers: architecture overview, key decisions + tradeoffs, post-deployment monitoring, and one production-only failure mode for a hybrid BM25 + dense vector RAG system serving ~2,000 documents to ~20 concurrent users on an air-gapped GPU cluster.

An Eraser.io architecture diagram source is embedded at the bottom of the document.

---

## Part B — Evaluation Harness

**Directory:** [part-b/](part-b/)  
**Full instructions:** [part-b/README.md](part-b/README.md)

### Quick start

```bash
cd part-b
uv venv && uv pip install -e ".[dev]"
source .venv/bin/activate
eval-harness sample_tests.jsonl --verbose
uv run pytest tests/ -v
```

Key design choices:
- **Zero runtime dependencies** — stdlib only (`difflib`, `json`, `urllib`)
- **Three scoring mechanisms**: exact match, keyword F1, sequence similarity
- **Mock endpoint modes**: `fixed`, `echo`, `random`, `fail`, `timeout`
- **CI-friendly**: exits with code 1 on any failure or error
- **JSON output mode** for machine consumption

---

## Part C — Opinion

**File:** [part-c/opinion.md](part-c/opinion.md)  
**PDF:** generated via `report-gen` (see below)

Three specific investigation strategies for stale/irrelevant answers 6 months post-deployment:
1. Silent re-indexing failures (md5sum reconciliation)
2. Retrieval recall regression (recall@5 evaluation against a golden set)
3. Chunk boundary fragmentation on updated documents

---

## Generating PDFs

PDFs for Parts A and C are produced with `md-to-pdf` (headless Chrome).

```bash
cd report-gen
pnpm install        # first time only
pnpm run build:all  # generates part-a/system-design.pdf + part-c/opinion.pdf
```

Individual builds:
```bash
pnpm run build:part-a   # part-a/system-design.pdf
pnpm run build:part-c   # part-c/opinion.pdf
```

Requires Node.js ≥ 18 and pnpm. The `report-gen/` directory contains the CSS stylesheet and `md-to-pdf` configuration originally set up for a prior project; only the build scripts and footer have been updated for this submission.
