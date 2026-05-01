"""
Interactive demo for the eval harness.

Run from the part-b directory:
    uv run python demo.py
    python demo.py  (with venv activated)

Walks through: loading test cases, scoring mechanics, mock endpoint modes,
and how score patterns map to Part C failure modes.
"""

import json
from harness.runner import load_test_cases, run
from harness.endpoint import MockEndpoint, HttpEndpoint
from harness.scorer import score as compute_score


DIVIDER = "=" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ── 1. Load test cases ────────────────────────────────────────────

section("1. Test cases (sample_tests.jsonl)")

cases = load_test_cases("sample_tests.jsonl")
print(f"Loaded {len(cases)} cases:\n")
for c in cases:
    print(f"  [{c.id}]  {c.input}")
    print(f"           expected: {c.expected}")


# ── 2. Scoring mechanics ─────────────────────────────────────────

section("2. Scoring mechanics")
print(
    "Three mechanisms, any one passing = PASS\n"
    "  exact_match       >= 1.0\n"
    "  keyword_f1        >= 0.6\n"
    "  sequence_similarity >= 0.7\n"
)

examples = [
    ("exact",      "14 days annual leave",                      "14 days annual leave"),
    ("paraphrase", "You are entitled to 14 days of annual leave.", "14 days annual leave"),
    ("partial",    "Employees receive leave each year.",         "14 days annual leave"),
    ("wrong",      "The policy is currently under review.",      "14 days annual leave"),
]

print(f"  {'Label':<12} {'EM':>5} {'KF1':>6} {'SS':>6}  Result")
print("  " + "-" * 50)
for label, response, expected in examples:
    s = compute_score("demo", response, expected)
    result = "PASS" if s.passed else "FAIL"
    print(f"  {label:<12} {s.exact_match:>5.2f} {s.keyword_f1:>6.2f} {s.sequence_similarity:>6.2f}  {result}")


# ── 3. Mock fixed endpoint (all fail) ────────────────────────────

section("3. Mock fixed endpoint — expected: all FAIL")

endpoint = MockEndpoint(mode="fixed")
summary = run("sample_tests.jsonl", endpoint, verbose=True)

print(f"\nPass rate: {summary.pass_rate:.1%}  ({summary.passed}/{summary.total})")
if summary.anomalies:
    print("\nAnomalies detected:")
    for a in summary.anomalies:
        print(f"  ! {a}")


# ── 4. Per-test score breakdown ──────────────────────────────────

section("4. Per-test score breakdown")
print(f"  {'ID':<6} {'EM':>5} {'KF1':>6} {'SS':>6}  {'Pass?'}")
print("  " + "-" * 36)
for r in summary.results:
    if r.score:
        s = r.score
        flag = "YES" if s.passed else "NO"
        print(f"  {r.test_id:<6} {s.exact_match:>5.2f} {s.keyword_f1:>6.2f} {s.sequence_similarity:>6.2f}  {flag}")
    else:
        print(f"  {r.test_id:<6}  ERROR: {r.error}")


# ── 5. Mock random endpoint ───────────────────────────────────────

section("5. Mock random endpoint — some may pass by chance")

endpoint = MockEndpoint(mode="random")
summary = run("sample_tests.jsonl", endpoint, verbose=True)
print(f"\nPass rate: {summary.pass_rate:.1%}  ({summary.passed}/{summary.total})")


# ── 6. Part C diagnostic guide ───────────────────────────────────

section("6. Part C diagnostic — how to read score patterns")

print("""
When running against the real Part A endpoint, score patterns map to
specific failure modes from Part C:

  keyword_f1 low on numeric queries (q3, q4, q5)
  → Chunk boundary fragmentation (Inv 3)
  → Fix: increase chunk_overlap 10% → 25%, rebuild indexes

  keyword_f1 drops on queries with new policy terminology
  after a monthly rebuild, while older queries still pass
  → BM25 vocabulary drift (Inv 2)
  → Fix: rebuild BM25 index, verify new terms are indexed

  All responses identical across different inputs
  → Endpoint anomaly (stuck vLLM / caching issue)
  → Check gateway logs

Run the harness before and after each monthly rebuild.
Compare JSON output across runs — a drop in pass rate or
keyword_f1 distribution narrows the root cause.
""")


# ── 7. Real endpoint template ─────────────────────────────────────

section("7. Real Part A endpoint (template)")

print("""
To run against the Part A FastAPI service:

    eval-harness sample_tests.jsonl \\
        --endpoint http://localhost:8080/generate \\
        --verbose

Or programmatically:

    endpoint = HttpEndpoint("http://localhost:8080/generate", timeout=30)
    summary = run("sample_tests.jsonl", endpoint, verbose=True)

The endpoint must accept POST {"prompt": "<query>"} and return one of:
    {"response": "..."}
    {"text": "..."}
    {"choices": [{"message": {"content": "..."}}]}  (OpenAI-style)
""")


# ── 8. JSON output ────────────────────────────────────────────────

section("8. JSON output (for CI or logging)")

endpoint = MockEndpoint(mode="fixed")
summary = run("sample_tests.jsonl", endpoint)

output = {
    "total": summary.total,
    "passed": summary.passed,
    "pass_rate": round(summary.pass_rate, 4),
    "anomalies": summary.anomalies,
    "results": [
        {
            "id": r.test_id,
            "passed": r.passed,
            "latency_ms": round(r.latency_ms, 1),
            "scores": {
                "exact_match": r.score.exact_match,
                "keyword_f1": round(r.score.keyword_f1, 4),
                "sequence_similarity": round(r.score.sequence_similarity, 4),
            } if r.score else None,
        }
        for r in summary.results
    ],
}

print(json.dumps(output, indent=2))
print("\nFrom CLI: eval-harness sample_tests.jsonl --output json > results.json")
