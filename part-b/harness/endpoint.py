"""
Endpoint abstractions for the eval harness.

MockEndpoint  — in-process stub; no network required.
KnowledgeBaseEndpoint — simulates a tiny RAG endpoint over a JSONL knowledge base.
HttpEndpoint  — calls a real HTTP LLM API (OpenAI-compatible or plain JSON).

All endpoint failures raise EndpointError so the runner can handle them
uniformly without catching urllib internals.
"""

import json
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


ABSTAIN_ANSWER = "I don't know based on the provided documents."
MIN_KEYWORD_OVERLAP = 2


@dataclass(frozen=True)
class EndpointResponse:
    answer: str
    sources: list[str]


class EndpointError(Exception):
    """Raised for any endpoint communication failure."""


class MockEndpoint:
    """
    In-process mock with configurable behaviour.

    Modes
    -----
    fixed   (default) — always returns the same string
    echo              — returns the prompt unchanged (tests prompt-leak detection)
    random            — returns a random subset of common HR keywords
    fail              — raises EndpointError (tests error handling)
    timeout           — raises EndpointError simulating a timeout
    """

    _HR_WORDS = [
        "policy", "approved", "manager", "14 days", "HR team",
        "direct report", "30 days", "extension", "reimbursement",
    ]

    def __init__(
        self,
        mode: str = "fixed",
        fixed_response: str = "This is a mock LLM response.",
    ) -> None:
        if mode not in {"fixed", "echo", "random", "fail", "timeout"}:
            raise ValueError(f"Unknown mock mode: {mode!r}")
        self.mode = mode
        self.fixed_response = fixed_response

    def call(self, prompt: str) -> EndpointResponse:
        if self.mode == "fixed":
            return EndpointResponse(answer=self.fixed_response, sources=[])
        if self.mode == "echo":
            return EndpointResponse(answer=prompt, sources=[])
        if self.mode == "random":
            k = random.randint(2, min(4, len(self._HR_WORDS)))
            return EndpointResponse(
                answer=" ".join(random.sample(self._HR_WORDS, k=k)),
                sources=[],
            )
        if self.mode == "fail":
            raise EndpointError("Simulated endpoint failure (mode=fail)")
        if self.mode == "timeout":
            raise EndpointError("Request timed out after 30s (mode=timeout)")
        raise AssertionError("unreachable")  # pragma: no cover


_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "at", "by", "for", "with", "what", "who", "when", "where", "how",
    "does", "do", "per",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if t not in _STOP}


class KnowledgeBaseEndpoint:
    """
    Simulates a tiny RAG endpoint over a JSONL knowledge base.

    Each line must contain:
      {"id": "...", "source": "...", "text": "...", "answer": "..."}

    The endpoint retrieves the record with the largest keyword overlap against
    the query, then returns its answer with a lightweight source citation.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._records: list[dict[str, str | set[str]]] = []
        parse_errors: list[str] = []

        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"Line {lineno}: invalid JSON - {exc}")
                    continue

                missing = [
                    k for k in ("id", "source", "text", "answer") if k not in obj
                ]
                if missing:
                    parse_errors.append(f"Line {lineno}: missing field(s) {missing}")
                    continue

                bad_fields = [
                    k for k in ("id", "source", "text", "answer")
                    if not isinstance(obj[k], str) or not obj[k].strip()
                ]
                if bad_fields:
                    parse_errors.append(
                        f"Line {lineno}: field(s) must be non-empty strings: {bad_fields}"
                    )
                    continue

                self._records.append(
                    {
                        "id": obj["id"],
                        "source": obj["source"],
                        "answer": obj["answer"],
                        "tokens": _tokens(
                            " ".join(
                                [obj["id"], obj["source"], obj["text"], obj["answer"]]
                            )
                        ),
                    }
                )

        if parse_errors:
            raise EndpointError(
                "Malformed knowledge base file:\n" + "\n".join(parse_errors)
            )
        if not self._records:
            raise EndpointError(f"Knowledge base file has no records: {path!r}")

    def call(self, prompt: str) -> EndpointResponse:
        query_tokens = _tokens(prompt)
        if not query_tokens:
            return EndpointResponse(answer=ABSTAIN_ANSWER, sources=[])

        best_record = None
        best_score = 0
        for record in self._records:
            score = len(query_tokens & record["tokens"])
            if score > best_score:
                best_score = score
                best_record = record

        # A single generic token match is too weak to count as grounded retrieval.
        if best_record is None or best_score < MIN_KEYWORD_OVERLAP:
            return EndpointResponse(answer=ABSTAIN_ANSWER, sources=[])

        return EndpointResponse(
            answer=str(best_record["answer"]),
            sources=[str(best_record["source"])],
        )


class HttpEndpoint:
    """
    Calls a real HTTP LLM endpoint.

    Handles OpenAI-style responses (`choices[0].message.content`),
    plain `response` fields, and `text` fields.
    """

    def __init__(
        self,
        url: str,
        timeout: int = 30,
        api_key: Optional[str] = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.api_key = api_key

    def call(self, prompt: str) -> EndpointResponse:
        payload = json.dumps({"prompt": prompt}).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.url, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise EndpointError(f"HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise EndpointError(f"Connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise EndpointError(f"Request timed out after {self.timeout}s") from exc
        except UnicodeDecodeError as exc:
            raise EndpointError("Endpoint returned non-UTF-8 JSON") from exc
        except json.JSONDecodeError as exc:
            raise EndpointError(f"Endpoint returned invalid JSON: {exc.msg}") from exc

        if not isinstance(data, dict):
            raise EndpointError(
                f"Endpoint returned unexpected JSON type: {type(data).__name__}"
            )

        # Normalise common LLM API response shapes
        if "answer" in data:
            return _build_response(data["answer"], data.get("sources", []))
        if "choices" in data:
            return _build_response(
                _extract_openai_content(data["choices"]),
                data.get("sources", []),
            )
        if "response" in data:
            return _build_response(data["response"], data.get("sources", []))
        if "text" in data:
            return _build_response(data["text"], data.get("sources", []))
        raise EndpointError(
            f"Unrecognised response shape; keys: {list(data.keys())}"
        )


def _build_response(answer: object, sources: object) -> EndpointResponse:
    if not isinstance(answer, str) or not answer.strip():
        raise EndpointError("Endpoint returned an empty or non-string answer")

    if sources is None:
        sources = []

    if not isinstance(sources, list) or any(
        not isinstance(source, str) or not source.strip() for source in sources
    ):
        raise EndpointError("Endpoint returned invalid sources; expected list[str]")

    return EndpointResponse(answer=answer, sources=sources)


def _extract_openai_content(choices: object) -> str:
    if not isinstance(choices, list) or not choices:
        raise EndpointError("Endpoint returned invalid OpenAI-style choices payload")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise EndpointError("Endpoint returned invalid OpenAI-style choices payload")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise EndpointError("Endpoint returned invalid OpenAI-style choices payload")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise EndpointError("Endpoint returned invalid OpenAI-style choices payload")

    return content
