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
    keyword_f1,
    sequence_similarity,
    semantic_sim,
    score,
    THRESHOLDS,
    _SEMANTIC_AVAILABLE,
)
from harness.runner import load_test_cases


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
        assert 0.0 <= result.keyword_f1 <= 1.0
        assert 0.0 <= result.sequence_similarity <= 1.0
        # semantic_sim is float if sentence-transformers installed, else None
        if result.semantic_sim is not None:
            assert 0.0 <= result.semantic_sim <= 1.0

    def test_both_empty(self):
        result = score("q1", "", "")
        assert result.passed


# ---------------------------------------------------------------------------
# semantic_sim
# ---------------------------------------------------------------------------


class TestSemanticSim:
    def test_returns_none_or_float(self):
        result = semantic_sim("annual leave policy", "leave entitlement")
        assert result is None or 0.0 <= result <= 1.0

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_identical_strings_score_near_one(self):
        s = semantic_sim("annual leave is 14 days", "annual leave is 14 days")
        assert s > 0.99

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_same_meaning_different_wording_passes_threshold(self):
        # "two weeks" vs "14 days": zero keyword overlap, same meaning.
        # all-MiniLM-L6-v2 scores this ~0.687, threshold is 0.65.
        s = semantic_sim("employees receive two weeks of annual leave", "14 days annual leave")
        assert s >= THRESHOLDS["semantic_sim"]

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_unrelated_strings_score_low(self):
        s = semantic_sim("the weather is sunny today", "annual leave is 14 days")
        assert s < THRESHOLDS["semantic_sim"]

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_empty_response_returns_zero(self):
        assert semantic_sim("", "14 days annual leave") == 0.0

    @pytest.mark.skipif(not _SEMANTIC_AVAILABLE, reason="sentence-transformers not installed")
    def test_score_passes_on_semantic_when_lexical_fails(self):
        # "two weeks" fails keyword_f1 and sequence_similarity against "14 days"
        # but semantic_sim should push it to PASS
        result = score("q1", "staff are entitled to two weeks of leave per year", "14 days annual leave")
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
