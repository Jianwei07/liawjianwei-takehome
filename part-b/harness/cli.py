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
import sys

from .endpoint import EndpointError, HttpEndpoint, KnowledgeBaseEndpoint, MockEndpoint
from .reporting import summary_to_json, summary_to_text
from .runner import run


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
  eval-harness data/tests_positive.jsonl --endpoint kb:data/knowledge_base.jsonl --verbose
  eval-harness data/tests_positive.jsonl --endpoint mock:random --verbose
  eval-harness data/tests_positive.jsonl --endpoint http://localhost:8080/generate --output json
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
        print(summary_to_json(summary))
    else:
        print(summary_to_text(summary))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(summary_to_json(summary))
        print(f"\nResults saved → {args.save}", file=sys.stderr)

    if summary.failed > 0 or summary.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
