"""
Unit tests for harness.scorer and harness.runner.load_test_cases.
"""

import json
import os
import tempfile

import pytest

from harness.scorer import (
    Score,
    normalize,
    exact_match,
    answer_coverage,
    keyword_f1,
    sequence_similarity,
    score,
    THRESHOLDS,
)
from harness.endpoint import MockEndpoint, KnowledgeBaseEndpoint
from harness.runner import load_test_cases, run


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases(self):
        assert normalize("Hello World") == "hello world"

    def test_strips_punctuation(self):
        result = normalize("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        assert normalize("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize("") == ""


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------


class TestExactMatch:
    def test_identical_strings(self):
        assert exact_match("14 days annual leave", "14 days annual leave") == 1.0

    def test_case_insensitive(self):
        assert exact_match("Direct Manager", "direct manager") == 1.0

    def test_different_strings(self):
        assert exact_match("14 days", "28 days") == 0.0

    def test_both_empty(self):
        assert exact_match("", "") == 1.0

    def test_one_empty(self):
        assert exact_match("", "something") == 0.0

    def test_punctuation_ignored(self):
        assert exact_match("14 days.", "14 days") == 1.0


# ---------------------------------------------------------------------------
# keyword_f1
# ---------------------------------------------------------------------------


class TestKeywordF1:
    def test_perfect_overlap(self):
        f1 = keyword_f1("annual leave 14 days", "annual leave 14 days")
        assert f1 == pytest.approx(1.0)

    def test_partial_recall(self):
        # Response has "direct manager" but expected also mentions "travel claims"
        f1 = keyword_f1("direct manager", "direct manager approves travel claims")
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        assert keyword_f1("banana apple orange", "direct manager travel") == pytest.approx(0.0)

    def test_superset_response(self):
        # All expected keywords present in a longer response
        f1 = keyword_f1("direct manager approves all expense claims", "direct manager")
        assert f1 > 0.0

    def test_stopword_only_expected(self):
        # Falls back gracefully when expected is all stopwords
        f1 = keyword_f1("this is a test", "a the is")
        assert 0.0 <= f1 <= 1.0

    def test_threshold_achievable(self):
        f1 = keyword_f1("14 days of annual leave per year", "14 days annual leave")
        assert f1 >= THRESHOLDS["keyword_f1"]


# ---------------------------------------------------------------------------
# answer_coverage
# ---------------------------------------------------------------------------


class TestAnswerCoverage:
    def test_verbose_correct_answer_covers_expected_keywords(self):
        coverage = answer_coverage(
            "According to the policy, employees receive 14 days annual leave per year.",
            "14 days annual leave",
        )
        assert coverage == pytest.approx(1.0)

    def test_missing_expected_keyword_is_partial(self):
        coverage = answer_coverage("Employees receive annual leave.", "14 days annual leave")
        assert 0.0 < coverage < 1.0


# ---------------------------------------------------------------------------
# sequence_similarity
# ---------------------------------------------------------------------------


class TestSequenceSimilarity:
    def test_identical(self):
        assert sequence_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        s = sequence_similarity("zzzzzz", "aaaaaa")
        assert s == pytest.approx(0.0)

    def test_close_strings(self):
        s = sequence_similarity(
            "14 days annual leave per year",
            "14 days annual leave",
        )
        assert s > 0.7

    def test_threshold_achievable(self):
        s = sequence_similarity("direct manager approves it", "direct manager")
        assert s >= THRESHOLDS["sequence_similarity"]


# ---------------------------------------------------------------------------
# score (integration)
# ---------------------------------------------------------------------------


class TestScore:
    def test_passes_on_exact_match(self):
        result = score("q1", "14 days annual leave", "14 days annual leave")
        assert result.passed
        assert result.reason == "exact match"

    def test_passes_on_keyword_f1(self):
        result = score(
            "q1",
            "employees receive 14 days of annual leave entitlement",
            "14 days annual leave",
        )
        assert result.passed

    def test_fails_completely_wrong(self):
        result = score("q1", "banana", "direct manager approves travel claims")
        assert not result.passed

    def test_failure_reason_contains_scores(self):
        result = score("q1", "wrong answer entirely", "correct answer here")
        if not result.passed:
            assert "exact=" in result.reason or "kf1=" in result.reason

    def test_returns_score_dataclass(self):
        result = score("q1", "response", "expected")
        assert isinstance(result, Score)
        assert result.test_id == "q1"
        assert 0.0 <= result.exact_match <= 1.0
        assert 0.0 <= result.answer_coverage <= 1.0
        assert 0.0 <= result.keyword_f1 <= 1.0
        assert 0.0 <= result.sequence_similarity <= 1.0

    def test_both_empty(self):
        result = score("q1", "", "")
        assert result.passed


# ---------------------------------------------------------------------------
# load_test_cases
# ---------------------------------------------------------------------------


class TestLoadTestCases:
    def _write_jsonl(self, lines: list[str]) -> str:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines))
        return path

    def test_valid_file(self):
        path = self._write_jsonl([
            '{"id": "q1", "input": "What is leave?", "expected": "14 days"}',
            '{"id": "q2", "input": "Who approves?", "expected": "Manager"}',
        ])
        cases = load_test_cases(path)
        assert len(cases) == 2
        assert cases[0].id == "q1"
        assert cases[1].expected == "Manager"

    def test_blank_lines_skipped(self):
        path = self._write_jsonl([
            '{"id": "q1", "input": "a", "expected": "b"}',
            "",
            '{"id": "q2", "input": "c", "expected": "d"}',
        ])
        cases = load_test_cases(path)
        assert len(cases) == 2

    def test_invalid_json_raises(self):
        path = self._write_jsonl(['{"id": "q1", bad json}'])
        with pytest.raises(ValueError, match="invalid JSON"):
            load_test_cases(path)

    def test_missing_fields_raises(self):
        path = self._write_jsonl(['{"id": "q1", "input": "hello"}'])
        with pytest.raises(ValueError, match="missing field"):
            load_test_cases(path)

    def test_non_string_fields_raise(self):
        path = self._write_jsonl([
            '{"id": 1, "input": "hello", "expected": "world"}',
        ])
        with pytest.raises(ValueError, match="non-empty strings"):
            load_test_cases(path)

    def test_empty_fields_raise(self):
        path = self._write_jsonl([
            '{"id": "q1", "input": "   ", "expected": "world"}',
        ])
        with pytest.raises(ValueError, match="non-empty strings"):
            load_test_cases(path)

    def test_multiple_errors_reported_together(self):
        path = self._write_jsonl([
            "not json at all",
            '{"id": "q2"}',  # missing input and expected
        ])
        with pytest.raises(ValueError) as exc_info:
            load_test_cases(path)
        msg = str(exc_info.value)
        assert "Line 1" in msg
        assert "Line 2" in msg

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_test_cases("/nonexistent/path/file.jsonl")


# ---------------------------------------------------------------------------
# runner + endpoint
# ---------------------------------------------------------------------------


class TestPipeline:
    def _write_jsonl(self, lines: list[str]) -> str:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines))
        return path

    def test_knowledge_base_endpoint_returns_sourced_answer(self):
        kb_path = self._write_jsonl([
            '{"id": "kb1", "source": "HR/leave.md", "text": "Annual leave policy gives 14 days annual leave.", "answer": "Employees receive 14 days annual leave."}',
        ])
        endpoint = KnowledgeBaseEndpoint(kb_path)
        response = endpoint.call("What is the annual leave policy?")
        assert "14 days annual leave" in response
        assert "Source: HR/leave.md" in response

    def test_knowledge_base_run_passes_verbose_answers(self):
        kb_path = self._write_jsonl([
            '{"id": "kb1", "source": "HR/leave.md", "text": "Annual leave policy gives 14 days annual leave.", "answer": "Employees receive 14 days annual leave."}',
            '{"id": "kb2", "source": "Finance/travel.md", "text": "Travel claims are approved by the direct manager.", "answer": "Travel claims are approved by the direct manager."}',
        ])
        path = self._write_jsonl([
            '{"id": "q1", "input": "What is the annual leave policy?", "expected": "14 days annual leave"}',
            '{"id": "q2", "input": "Who approves travel claims?", "expected": "Direct manager"}',
        ])
        summary = run(path, KnowledgeBaseEndpoint(kb_path))
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.errors == 0

    def test_endpoint_failure_records_errors(self):
        path = self._write_jsonl([
            '{"id": "q1", "input": "What is leave?", "expected": "14 days"}',
        ])
        summary = run(path, MockEndpoint(mode="fail"))
        assert summary.total == 1
        assert summary.errors == 1
        assert "Simulated endpoint failure" in summary.results[0].error

    def test_fixed_mock_triggers_identical_response_anomaly(self):
        path = self._write_jsonl([
            '{"id": "q1", "input": "a", "expected": "x"}',
            '{"id": "q2", "input": "b", "expected": "y"}',
        ])
        summary = run(path, MockEndpoint(mode="fixed"))
        assert summary.failed == 2
        assert any("identical" in anomaly for anomaly in summary.anomalies)
