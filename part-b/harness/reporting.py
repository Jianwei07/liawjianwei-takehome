"""Helpers for rendering eval summaries as text or JSON payloads."""

import json

from .runner import RunSummary, TestResult


def _score_payload(result: TestResult) -> dict[str, float] | None:
    if result.score is None:
        return None

    return {
        "exact_match": result.score.exact_match,
        "answer_coverage": round(result.score.answer_coverage, 4),
        "keyword_f1": round(result.score.keyword_f1, 4),
        "sequence_similarity": round(result.score.sequence_similarity, 4),
    }


def _result_payload(result: TestResult) -> dict[str, object]:
    return {
        "id": result.test_id,
        "case_type": result.case_type,
        "tags": result.tags,
        "passed": result.passed,
        "failure_bucket": result.failure_bucket,
        "answer": result.answer,
        "sources": result.sources,
        "expected_answer": result.expected_answer,
        "expected_sources": result.expected_sources,
        "error": result.error,
        "latency_ms": round(result.latency_ms, 1),
        "source_match": result.source_match,
        "abstained": result.abstained,
        "scores": _score_payload(result),
        "reason": result.reason,
    }


def summary_to_payload(summary: RunSummary) -> dict[str, object]:
    return {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "errors": summary.errors,
        "pass_rate": round(summary.pass_rate, 4),
        "grounding_cases": summary.grounding_cases,
        "grounding_rate": round(summary.grounding_rate, 4),
        "abstention_cases": summary.abstention_cases,
        "abstention_success_rate": round(summary.abstention_success_rate, 4),
        "failure_buckets": summary.failure_buckets,
        "anomalies": summary.anomalies,
        "results": [_result_payload(result) for result in summary.results],
    }


def summary_to_json(summary: RunSummary) -> str:
    return json.dumps(summary_to_payload(summary), indent=2)


def summary_to_text(summary: RunSummary) -> str:
    lines = [
        "=" * 52,
        "EVAL HARNESS — SUMMARY",
        "=" * 52,
        f"Total:            {summary.total}",
        f"Passed:           {summary.passed}",
        f"Failed:           {summary.failed}",
        f"Errors:           {summary.errors}",
        f"Pass rate:        {summary.pass_rate:.1%}",
    ]

    if summary.grounding_cases:
        lines.append(
            f"Grounding rate:   {summary.grounding_rate:.1%} "
            f"({summary.grounding_cases} source-checked cases)"
        )
    if summary.abstention_cases:
        lines.append(
            f"Abstain success:  {summary.abstention_success_rate:.1%} "
            f"({summary.abstention_cases} abstain cases)"
        )

    if summary.failure_buckets:
        lines.append("\nFAILURE BUCKETS:")
        for bucket, count in sorted(summary.failure_buckets.items()):
            lines.append(f"  {bucket}: {count}")

    if summary.anomalies:
        lines.append("\nANOMALIES DETECTED:")
        for anomaly in summary.anomalies:
            lines.append(f"  ! {anomaly}")

    failures = [result for result in summary.results if not result.passed]
    if failures:
        lines.append("\nFAILURES:")
        for result in failures:
            bucket = result.failure_bucket or "failed"
            lines.append(f"  [{result.test_id}] {bucket}")
            if result.error:
                lines.append(f"    Error:   {result.error}")
                continue
            lines.append(f"    Answer:  {result.answer!r}")
            if result.sources:
                lines.append(f"    Sources: {result.sources}")
            if result.expected_answer is not None:
                lines.append(f"    Expect:  {result.expected_answer!r}")
            if result.expected_sources:
                lines.append(f"    Cites:   {result.expected_sources}")
            lines.append(f"    Reason:  {result.reason}")

    return "\n".join(lines)
