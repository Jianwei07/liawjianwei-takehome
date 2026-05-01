"""
Test runner — loads JSONL test cases, calls the endpoint, scores responses,
detects anomalies, and returns a structured RunSummary.
"""

from collections import Counter
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .endpoint import ABSTAIN_ANSWER, EndpointError, EndpointResponse
from .scorer import Score, normalize, score as compute_score


@dataclass
class TestCase:
    id: str
    input: str
    case_type: str
    expected_answer: Optional[str] = None
    expected_sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    test_id: str
    case_type: str
    tags: list[str]
    expected_answer: Optional[str]
    expected_sources: list[str]
    answer: Optional[str]
    sources: list[str]
    score: Optional[Score]
    error: Optional[str]
    latency_ms: float
    passed: bool
    failure_bucket: Optional[str]
    reason: str
    source_match: Optional[bool]
    abstained: Optional[bool]


@dataclass
class RunSummary:
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    grounding_cases: int
    grounding_rate: float
    abstention_cases: int
    abstention_success_rate: float
    failure_buckets: dict[str, int]
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

            missing = [k for k in ("id", "input") if k not in obj]
            if missing:
                parse_errors.append(
                    f"Line {lineno}: missing field(s) {missing}"
                )
                continue

            bad_fields = [
                k for k in ("id", "input")
                if not isinstance(obj[k], str) or not obj[k].strip()
            ]
            if bad_fields:
                parse_errors.append(
                    f"Line {lineno}: field(s) must be non-empty strings: {bad_fields}"
                )
                continue

            case_type = obj.get("case_type", "answer")
            if case_type not in {"answer", "abstain"}:
                parse_errors.append(
                    f"Line {lineno}: case_type must be 'answer' or 'abstain'"
                )
                continue

            expected_answer = obj.get("expected_answer", obj.get("expected"))
            if case_type == "answer" and (
                not isinstance(expected_answer, str) or not expected_answer.strip()
            ):
                parse_errors.append(
                    f"Line {lineno}: answer cases require a non-empty expected_answer"
                )
                continue

            expected_sources = _parse_string_list(
                obj.get("expected_sources", []),
                field_name="expected_sources",
                lineno=lineno,
                parse_errors=parse_errors,
            )
            tags = _parse_string_list(
                obj.get("tags", []),
                field_name="tags",
                lineno=lineno,
                parse_errors=parse_errors,
            )
            if expected_sources is None or tags is None:
                continue

            cases.append(
                TestCase(
                    id=obj["id"],
                    input=obj["input"],
                    case_type=case_type,
                    expected_answer=expected_answer.strip()
                    if isinstance(expected_answer, str)
                    else None,
                    expected_sources=expected_sources,
                    tags=tags,
                )
            )

    if parse_errors:
        raise ValueError("Malformed test file:\n" + "\n".join(parse_errors))

    return cases


def _detect_anomalies(results: list[TestResult]) -> list[str]:
    anomalies: list[str] = []

    empty = [r.test_id for r in results if r.answer == ""]
    if empty:
        anomalies.append(f"Empty responses for: {empty}")

    # All non-error responses identical → endpoint may be stuck
    non_error = [
        r.answer
        for r in results
        if r.answer is not None and r.abstained is not True
    ]
    if len(non_error) > 1 and len(set(non_error)) == 1:
        anomalies.append(
            f"All {len(non_error)} responses are identical — endpoint may be stuck"
        )

    slow = [r.test_id for r in results if r.latency_ms > 10_000]
    if slow:
        anomalies.append(f"Slow responses (>10 s) for: {slow}")

    return anomalies


def _parse_string_list(
    value: object,
    *,
    field_name: str,
    lineno: int,
    parse_errors: list[str],
) -> list[str] | None:
    if value is None:
        return []

    if not isinstance(value, list):
        parse_errors.append(f"Line {lineno}: {field_name} must be a list of strings")
        return None

    bad_items = [item for item in value if not isinstance(item, str) or not item.strip()]
    if bad_items:
        parse_errors.append(
            f"Line {lineno}: {field_name} must contain non-empty strings"
        )
        return None

    return [item.strip() for item in value]


def _source_match(actual_sources: list[str], expected_sources: list[str]) -> bool:
    actual = {source.strip() for source in actual_sources}
    return any(source in actual for source in expected_sources)


def _evaluate_case(
    case: TestCase,
    response: EndpointResponse,
) -> tuple[Optional[Score], bool, Optional[str], str, Optional[bool], Optional[bool]]:
    if case.case_type == "abstain":
        abstained = normalize(response.answer) == normalize(ABSTAIN_ANSWER)
        if abstained:
            return None, True, None, "abstained as expected", None, True
        return (
            None,
            False,
            "abstain_miss",
            f"expected abstain, got {response.answer!r}",
            None,
            False,
        )

    assert case.expected_answer is not None
    score = compute_score(case.id, response.answer, case.expected_answer)
    source_match = (
        _source_match(response.sources, case.expected_sources)
        if case.expected_sources
        else None
    )

    if not score.passed:
        return score, False, "answer_mismatch", score.reason, source_match, None

    if case.expected_sources and not source_match:
        return (
            score,
            False,
            "source_mismatch",
            f"expected one of {case.expected_sources!r}, got {response.sources!r}",
            source_match,
            None,
        )

    reason = score.reason
    if case.expected_sources:
        reason = f"{reason}; source match"
    return score, True, None, reason, source_match, None


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
            score, passed, failure_bucket, reason, source_match, abstained = _evaluate_case(
                case,
                response,
            )
            result = TestResult(
                test_id=case.id,
                case_type=case.case_type,
                tags=case.tags,
                expected_answer=case.expected_answer,
                expected_sources=case.expected_sources,
                answer=response.answer,
                sources=response.sources,
                score=score,
                error=None,
                latency_ms=latency_ms,
                passed=passed,
                failure_bucket=failure_bucket,
                reason=reason,
                source_match=source_match,
                abstained=abstained,
            )
        except EndpointError as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            result = TestResult(
                test_id=case.id,
                case_type=case.case_type,
                tags=case.tags,
                expected_answer=case.expected_answer,
                expected_sources=case.expected_sources,
                answer=None,
                sources=[],
                score=None,
                error=str(exc),
                latency_ms=latency_ms,
                passed=False,
                failure_bucket="endpoint_error",
                reason=str(exc),
                source_match=None,
                abstained=None,
            )

        results.append(result)

        if verbose:
            if result.error:
                label = "ERROR"
            elif result.passed:
                label = "PASS"
            else:
                label = "FAIL"
            detail = f"  [{label}] {case.id}"
            if result.failure_bucket:
                detail += f" ({result.failure_bucket})"
            print(detail)

    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.failure_bucket == "endpoint_error")
    failed = len(results) - passed - errors
    failure_buckets = dict(
        sorted(
            Counter(
                r.failure_bucket for r in results if r.failure_bucket is not None
            ).items()
        )
    )
    grounding_results = [
        r for r in results if r.case_type == "answer" and r.source_match is not None
    ]
    grounding_matches = sum(1 for r in grounding_results if r.source_match)
    abstention_results = [r for r in results if r.case_type == "abstain"]
    abstention_matches = sum(1 for r in abstention_results if r.abstained)

    return RunSummary(
        total=len(results),
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=passed / len(results) if results else 0.0,
        grounding_cases=len(grounding_results),
        grounding_rate=(
            grounding_matches / len(grounding_results) if grounding_results else 0.0
        ),
        abstention_cases=len(abstention_results),
        abstention_success_rate=(
            abstention_matches / len(abstention_results)
            if abstention_results
            else 0.0
        ),
        failure_buckets=failure_buckets,
        results=results,
        anomalies=_detect_anomalies(results),
    )
