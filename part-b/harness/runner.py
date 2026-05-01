"""
Test runner — loads JSONL test cases, calls the endpoint, scores responses,
detects anomalies, and returns a structured RunSummary.
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

from .endpoint import EndpointError
from .scorer import Score, score as compute_score


@dataclass
class TestCase:
    id: str
    input: str
    expected: str


@dataclass
class TestResult:
    test_id: str
    response: Optional[str]
    score: Optional[Score]
    error: Optional[str]
    latency_ms: float

    @property
    def passed(self) -> bool:
        return self.score is not None and self.score.passed


@dataclass
class RunSummary:
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    results: list[TestResult]
    anomalies: list[str]


def load_test_cases(path: str) -> list[TestCase]:
    """
    Parse a JSONL file into TestCase objects.

    Collects ALL parse errors before raising so the caller sees the full
    picture in one shot rather than discovering errors one at a time.
    """
    cases: list[TestCase] = []
    parse_errors: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(f"Line {lineno}: invalid JSON — {exc}")
                continue

            missing = [k for k in ("id", "input", "expected") if k not in obj]
            if missing:
                parse_errors.append(
                    f"Line {lineno}: missing field(s) {missing}"
                )
                continue

            bad_fields = [
                k for k in ("id", "input", "expected")
                if not isinstance(obj[k], str) or not obj[k].strip()
            ]
            if bad_fields:
                parse_errors.append(
                    f"Line {lineno}: field(s) must be non-empty strings: {bad_fields}"
                )
                continue

            cases.append(
                TestCase(id=obj["id"], input=obj["input"], expected=obj["expected"])
            )

    if parse_errors:
        raise ValueError("Malformed test file:\n" + "\n".join(parse_errors))

    return cases


def _detect_anomalies(results: list[TestResult]) -> list[str]:
    anomalies: list[str] = []

    empty = [r.test_id for r in results if r.response == ""]
    if empty:
        anomalies.append(f"Empty responses for: {empty}")

    # All non-error responses identical → endpoint may be stuck
    non_error = [r.response for r in results if r.response is not None]
    if len(non_error) > 1 and len(set(non_error)) == 1:
        anomalies.append(
            f"All {len(non_error)} responses are identical — endpoint may be stuck"
        )

    slow = [r.test_id for r in results if r.latency_ms > 10_000]
    if slow:
        anomalies.append(f"Slow responses (>10 s) for: {slow}")

    return anomalies


def run(
    test_file: str,
    endpoint,
    verbose: bool = False,
) -> RunSummary:
    cases = load_test_cases(test_file)
    results: list[TestResult] = []

    for case in cases:
        t0 = time.monotonic()
        try:
            response = endpoint.call(case.input)
            latency_ms = (time.monotonic() - t0) * 1000
            s = compute_score(case.id, response, case.expected)
            result = TestResult(
                test_id=case.id,
                response=response,
                score=s,
                error=None,
                latency_ms=latency_ms,
            )
        except EndpointError as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            result = TestResult(
                test_id=case.id,
                response=None,
                score=None,
                error=str(exc),
                latency_ms=latency_ms,
            )

        results.append(result)

        if verbose:
            if result.error:
                label = "ERROR"
            elif result.passed:
                label = "PASS"
            else:
                label = "FAIL"
            print(f"  [{label}] {case.id}")

    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error is not None)
    failed = len(results) - passed - errors

    return RunSummary(
        total=len(results),
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=passed / len(results) if results else 0.0,
        results=results,
        anomalies=_detect_anomalies(results),
    )
