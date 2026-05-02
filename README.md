# AI Engineer Take-Home — Liaw Jian Wei (Jayden)
---

## Repo Structure

```text
.
├── part-a/               Part A: system design
│   ├── system-design.md
│   └── system-design.pdf
├── part-b/               Part B: grounded evaluation service
│   ├── data/             Knowledge base + eval suites
│   ├── harness/          Runner, scorer, endpoint adapters, reporting
│   ├── tests/            Backend and scoring tests
│   ├── main.py           HTTP service entrypoint
│   ├── pyproject.toml
│   └── README.md         Start here for Part B
└── part-c/               Part C: opinion writeup
    ├── opinion.md
    └── opinion.pdf
```

---

## Part A — System Design

- Markdown: [part-a/system-design.md](part-a/system-design.md)
- PDF: [part-a/system-design.pdf](part-a/system-design.pdf)

Covers:

- architecture overview
- key decisions and tradeoffs
- post-deployment monitoring
- one production-only failure mode

The design targets a hybrid BM25 + dense retrieval RAG system serving roughly
2,000 internal documents to about 20 concurrent users in an air-gapped
environment.

---

## Part B — Grounded Evaluation Service

- Directory: [part-b/](part-b/)
- Full instructions: [part-b/README.md](part-b/README.md)

### Quick Start

```bash
cd part-b
uv sync --extra dev
uv run pytest tests/ -v
uv run task dev
```

In another terminal:

```bash
PART_B_DIR="$(git rev-parse --show-toplevel)/part-b"

curl -sS -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the annual leave policy?"}'

curl -sS -X POST http://127.0.0.1:8080/eval \
  -o "$PART_B_DIR/positive.json"
```

Highlights:

- service-only implementation, no separate CLI path
- grounded answer contract: `answer` + `sources`
- positive and diagnostic JSONL suites
- abstention on unknown questions
- structured eval summaries with failure buckets

---

## Part C — Opinion

- Markdown: [part-c/opinion.md](part-c/opinion.md)
- PDF: [part-c/opinion.pdf](part-c/opinion.pdf)

Focus areas:

1. silent re-indexing failures
2. retrieval recall regression
3. chunk-boundary fragmentation after document updates

---

## Notes

Personal notes, local tooling, and the assignment brief are intentionally not
tracked in the repository.
