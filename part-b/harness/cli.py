"""
CLI entry point for the LLM eval harness.

Usage
-----
  eval-harness <test_file> [options]

Exit codes
----------
  0  all tests passed
  1  one or more failures or errors (CI-friendly)
"""

import argparse
import json
import sys

from .endpoint import EndpointError, HttpEndpoint, KnowledgeBaseEndpoint, MockEndpoint
from .runner import run, RunSummary


def _text_summary(summary: RunSummary) -> str:
    lines = [
        "=" * 52,
        "EVAL HARNESS — SUMMARY",
        "=" * 52,
        f"Total:     {summary.total}",
        f"Passed:    {summary.passed}",
        f"Failed:    {summary.failed}",
        f"Errors:    {summary.errors}",
        f"Pass rate: {summary.pass_rate:.1%}",
    ]

    if summary.anomalies:
        lines.append("\nANOMALIES DETECTED:")
        for a in summary.anomalies:
            lines.append(f"  ! {a}")

    failures = [r for r in summary.results if not r.passed]
    if failures:
        lines.append("\nFAILURES:")
        for r in failures:
            lines.append(f"  [{r.test_id}]")
            if r.error:
                lines.append(f"    Error:    {r.error}")
            else:
                lines.append(f"    Response: {r.response!r}")
                lines.append(f"    Reason:   {r.score.reason}")

    return "\n".join(lines)


def _json_summary(summary: RunSummary) -> str:
    payload = {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "errors": summary.errors,
        "pass_rate": round(summary.pass_rate, 4),
        "anomalies": summary.anomalies,
        "results": [
            {
                "id": r.test_id,
                "passed": r.passed,
                "response": r.response,
                "error": r.error,
                "latency_ms": round(r.latency_ms, 1),
                "scores": {
                    "exact_match": r.score.exact_match,
                    "answer_coverage": round(r.score.answer_coverage, 4),
                    "keyword_f1": round(r.score.keyword_f1, 4),
                    "sequence_similarity": round(r.score.sequence_similarity, 4),
                }
                if r.score
                else None,
                "reason": r.score.reason if r.score else r.error,
            }
            for r in summary.results
        ],
    }
    return json.dumps(payload, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval-harness",
        description="Run LLM evaluation tests from a JSONL file against an endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
endpoint formats:
  mock               fixed stub response (default)
  mock:echo          echoes the input prompt back
  mock:random        random HR-domain keywords
  mock:fail          simulates endpoint failure
  mock:timeout       simulates request timeout
  kb:path.jsonl      simulate a RAG endpoint over a JSONL knowledge base
  http://host/path   real HTTP endpoint (POST, JSON body)
  https://host/path  real HTTPS endpoint

examples:
  eval-harness sample_tests.jsonl --endpoint kb:sample_knowledge_base.jsonl --verbose
  eval-harness sample_tests.jsonl --endpoint mock:random --verbose
  eval-harness sample_tests.jsonl --endpoint http://localhost:8080/generate --output json
        """,
    )
    parser.add_argument("test_file", help="Path to JSONL test file")
    parser.add_argument(
        "--endpoint",
        default="mock",
        metavar="ENDPOINT",
        help="Endpoint to call (default: mock)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token for HTTP endpoints",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-test PASS/FAIL/ERROR as tests run",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        nargs="?",
        const="results.json",
        default=None,
        help="Save full JSON results to FILE (default: results.json)",
    )

    args = parser.parse_args()

    # Build endpoint
    ep_str: str = args.endpoint
    if ep_str.startswith("mock"):
        mode = ep_str.split(":", 1)[1] if ":" in ep_str else "fixed"
        try:
            endpoint = MockEndpoint(mode=mode)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    elif ep_str.startswith("kb:"):
        kb_path = ep_str.split(":", 1)[1]
        try:
            endpoint = KnowledgeBaseEndpoint(kb_path)
        except (FileNotFoundError, EndpointError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    elif ep_str.startswith(("http://", "https://")):
        endpoint = HttpEndpoint(ep_str, timeout=args.timeout, api_key=args.api_key)
    else:
        print(
            f"Error: unrecognised endpoint {ep_str!r}. "
            "Use 'mock', 'mock:<mode>', 'kb:<path>', or an http(s):// URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        summary = run(args.test_file, endpoint, verbose=args.verbose)
    except FileNotFoundError:
        print(f"Error: test file not found: {args.test_file!r}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        print(_json_summary(summary))
    else:
        print(_text_summary(summary))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(_json_summary(summary))
        print(f"\nResults saved → {args.save}", file=sys.stderr)

    if summary.failed > 0 or summary.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
