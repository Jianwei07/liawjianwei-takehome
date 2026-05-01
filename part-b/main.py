"""
Tiny local HTTP endpoint for exercising the eval harness.

This is intentionally small: it wraps the existing KnowledgeBaseEndpoint behind
two curl-able routes without adding a web framework dependency:

  POST /generate  answer one query
  POST /eval      run the configured JSONL tests and return the eval summary
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from harness.endpoint import EndpointError, KnowledgeBaseEndpoint
from harness.reporting import summary_to_payload
from harness.runner import run


_PART_B_DIR = Path(__file__).resolve().parent


def _resolve_kb_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return str(candidate)

    beside_script = Path(__file__).resolve().parent / path
    if beside_script.is_file():
        return str(beside_script)

    return str(candidate)


def _resolve_eval_test_path(path: str, *, base_dir: Path) -> str:
    raw_path = Path(path).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [
        _PART_B_DIR / raw_path,
        base_dir / raw_path,
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(base_dir)
        except ValueError:
            continue

        if resolved.suffix != ".jsonl":
            raise ValueError("tests must point to a .jsonl file")

        return str(resolved)

    raise ValueError("tests must resolve inside the configured test directory")


class HarnessHandler(BaseHTTPRequestHandler):
    endpoint: KnowledgeBaseEndpoint
    test_file: str
    server_version = "EvalHarnessEndpoint/0.1"

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/eval":
            self._handle_eval()
            return
        if path != "/generate":
            self._send_json(404, {"error": "Not found"})
            return

        self._handle_generate()

    def _read_json_body(self) -> dict[str, object] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return None

        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8") if raw_body else "{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc.msg}"})
            return
        except UnicodeDecodeError:
            self._send_json(400, {"error": "Request body must be valid UTF-8 JSON"})
            return

        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return None

        return payload

    def _handle_generate(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return

        prompt = payload.get("prompt") or payload.get("input")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(400, {"error": "Request must include a non-empty prompt"})
            return

        try:
            response = self.endpoint.call(prompt)
        except EndpointError as exc:
            self._send_json(500, {"error": str(exc)})
            return

        self._send_json(200, {"answer": response.answer, "sources": response.sources})

    def _handle_eval(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return

        test_file = self.test_file
        if "tests" in payload:
            requested_tests = payload["tests"]
            if not isinstance(requested_tests, str) or not requested_tests.strip():
                self._send_json(400, {"error": "tests must be a non-empty string"})
                return
            try:
                test_file = _resolve_eval_test_path(
                    requested_tests,
                    base_dir=Path(self.test_file).resolve().parent,
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

        try:
            summary = run(test_file, self.endpoint)
        except FileNotFoundError:
            self._send_json(
                500,
                {"error": f"Test file not found: {test_file!r}"},
            )
            return
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        payload = summary_to_payload(summary)
        payload["test_file"] = test_file
        self._send_json(200, payload)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny local dummy LLM/RAG endpoint for the eval harness."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind (default: 8080)",
    )
    parser.add_argument(
        "--kb",
        default="data/knowledge_base.jsonl",
        help="JSONL knowledge base path (default: data/knowledge_base.jsonl)",
    )
    parser.add_argument(
        "--tests",
        default="data/tests_positive.jsonl",
        help="JSONL test file path for POST /eval (default: data/tests_positive.jsonl)",
    )
    args = parser.parse_args()

    kb_path = _resolve_kb_path(args.kb)
    test_file = _resolve_kb_path(args.tests)
    try:
        HarnessHandler.endpoint = KnowledgeBaseEndpoint(kb_path)
        HarnessHandler.test_file = test_file
    except (FileNotFoundError, EndpointError) as exc:
        print(f"Error loading knowledge base: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        server = ThreadingHTTPServer((args.host, args.port), HarnessHandler)
    except OSError as exc:
        print(f"Error starting endpoint: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Serving eval endpoint at http://{args.host}:{args.port}", flush=True)
    print("Routes: GET /health, POST /generate, POST /eval", flush=True)
    print(f"Knowledge base: {kb_path}", flush=True)
    print(f"Test file: {test_file}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down eval endpoint")
    finally:
        server.server_close()


def task() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "dev":
        sys.argv = [sys.argv[0], "--port", "8080"]
        main()
        return

    print("Usage: uv run task dev", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
