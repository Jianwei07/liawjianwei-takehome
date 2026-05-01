import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from harness.endpoint import ABSTAIN_ANSWER, KnowledgeBaseEndpoint
from main import HarnessHandler


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _test_server(kb_path: Path, test_path: Path):
    class TestHandler(HarnessHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

    TestHandler.endpoint = KnowledgeBaseEndpoint(str(kb_path))
    TestHandler.test_file = str(test_path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post_json(url: str, payload: dict | None = None) -> dict:
    data = b"" if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def test_backend_routes_run_full_eval_flow(tmp_path: Path):
    kb_path = tmp_path / "kb.jsonl"
    test_path = tmp_path / "tests.jsonl"
    diagnostic_test_path = tmp_path / "tests_diagnostic.jsonl"
    _write_jsonl(
        kb_path,
        [
            {
                "id": "travel_policy",
                "source": "Finance/2026/travel-claims.md",
                "text": "Travel claims must be approved by the direct manager before reimbursement.",
                "answer": "Travel claims are approved by the direct manager.",
            },
            {
                "id": "travel_faq",
                "source": "Finance/2026/travel-faq.md",
                "text": "For domestic travel, claims are approved by the direct manager. This FAQ summarises the policy.",
                "answer": "Domestic travel claims are approved by the direct manager.",
            },
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "id": "p1",
                "input": "Who approves travel claims before reimbursement?",
                "case_type": "answer",
                "expected_answer": "Direct manager",
                "expected_sources": ["Finance/2026/travel-claims.md"],
            },
        ],
    )
    _write_jsonl(
        diagnostic_test_path,
        [
            {
                "id": "d1",
                "input": "Which manager approves domestic travel claims?",
                "case_type": "answer",
                "expected_answer": "Direct manager",
                "expected_sources": ["Finance/2026/travel-claims.md"],
            },
        ],
    )

    with _test_server(kb_path, test_path) as base_url:
        health = _get_json(f"{base_url}/health")
        generated = _post_json(
            f"{base_url}/generate",
            {"prompt": "Who approves travel claims before reimbursement?"},
        )
        summary = _post_json(f"{base_url}/eval")
        diagnostic_summary = _post_json(
            f"{base_url}/eval",
            {"tests": str(diagnostic_test_path)},
        )

    assert health == {"status": "ok"}
    assert generated["answer"] == "Travel claims are approved by the direct manager."
    assert generated["sources"] == ["Finance/2026/travel-claims.md"]
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["grounding_rate"] == 1.0
    assert summary["failure_buckets"] == {}
    assert diagnostic_summary["total"] == 1
    assert diagnostic_summary["passed"] == 0
    assert diagnostic_summary["failed"] == 1
    assert diagnostic_summary["failure_buckets"] == {"source_mismatch": 1}
    assert diagnostic_summary["results"][0]["sources"] == [
        "Finance/2026/travel-faq.md"
    ]


def test_backend_generate_returns_abstain_for_unknown_query(tmp_path: Path):
    kb_path = tmp_path / "kb.jsonl"
    test_path = tmp_path / "tests.jsonl"
    _write_jsonl(
        kb_path,
        [
            {
                "id": "kb_leave",
                "source": "HR/leave.md",
                "text": "Annual leave policy gives 14 days annual leave.",
                "answer": "Employees receive 14 days annual leave.",
            },
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "id": "p1",
                "input": "What is the annual leave policy?",
                "expected_answer": "14 days annual leave",
                "expected_sources": ["HR/leave.md"],
            },
        ],
    )

    with _test_server(kb_path, test_path) as base_url:
        generated = _post_json(
            f"{base_url}/generate",
            {"prompt": "What is the parking reimbursement policy?"},
        )

    assert generated == {"answer": ABSTAIN_ANSWER, "sources": []}


def test_backend_generate_rejects_empty_prompt(tmp_path: Path):
    kb_path = tmp_path / "kb.jsonl"
    test_path = tmp_path / "tests.jsonl"
    _write_jsonl(
        kb_path,
        [
            {
                "id": "kb_leave",
                "source": "HR/leave.md",
                "text": "Annual leave policy gives 14 days annual leave.",
                "answer": "Employees receive 14 days annual leave.",
            },
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "id": "p1",
                "input": "What is the annual leave policy?",
                "expected_answer": "14 days annual leave",
                "expected_sources": ["HR/leave.md"],
            },
        ],
    )

    with _test_server(kb_path, test_path) as base_url:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(f"{base_url}/generate", {"prompt": ""})

        payload = json.loads(exc_info.value.read())

    assert exc_info.value.code == 400
    assert "prompt" in payload["error"]


def test_backend_rejects_non_object_json_body(tmp_path: Path):
    kb_path = tmp_path / "kb.jsonl"
    test_path = tmp_path / "tests.jsonl"
    _write_jsonl(
        kb_path,
        [
            {
                "id": "kb_leave",
                "source": "HR/leave.md",
                "text": "Annual leave policy gives 14 days annual leave.",
                "answer": "Employees receive 14 days annual leave.",
            },
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "id": "p1",
                "input": "What is the annual leave policy?",
                "expected_answer": "14 days annual leave",
                "expected_sources": ["HR/leave.md"],
            },
        ],
    )

    with _test_server(kb_path, test_path) as base_url:
        req = urllib.request.Request(
            f"{base_url}/generate",
            data=b"[]",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)

        payload = json.loads(exc_info.value.read())

    assert exc_info.value.code == 400
    assert payload["error"] == "JSON body must be an object"


def test_backend_eval_rejects_test_override_outside_part_b(tmp_path: Path):
    kb_path = tmp_path / "kb.jsonl"
    test_path = tmp_path / "tests.jsonl"
    _write_jsonl(
        kb_path,
        [
            {
                "id": "kb_leave",
                "source": "HR/leave.md",
                "text": "Annual leave policy gives 14 days annual leave.",
                "answer": "Employees receive 14 days annual leave.",
            },
        ],
    )
    _write_jsonl(
        test_path,
        [
            {
                "id": "p1",
                "input": "What is the annual leave policy?",
                "expected_answer": "14 days annual leave",
                "expected_sources": ["HR/leave.md"],
            },
        ],
    )

    with _test_server(kb_path, test_path) as base_url:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(
                f"{base_url}/eval",
                {"tests": "../outside.jsonl"},
            )

        payload = json.loads(exc_info.value.read())

    assert exc_info.value.code == 400
    assert "configured test directory" in payload["error"]
