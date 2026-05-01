"""
Scoring module for the LLM eval harness.

Four mechanisms, all stdlib, no ML dependencies required:

  1. exact_match - normalized string equality (binary 1.0 / 0.0)
  2. answer_coverage - expected keywords covered by the response
  3. keyword_f1 - precision/recall/F1 over non-stopword token sets
  4. sequence_similarity - difflib SequenceMatcher ratio

A test case passes when ANY mechanism meets its threshold:
  exact_match >= 1.0 OR answer_coverage >= 1.0 OR keyword_f1 >= 0.6
  OR sequence_similarity >= 0.7

Thresholds are simple starting values: permissive enough for natural-language
paraphrases, but still strict enough to reject clearly wrong answers.
"""

import difflib
import re
from dataclasses import dataclass


# Stopwords excluded from keyword_f1 to avoid penalising phrasing differences.
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "as", "into", "that", "this", "it", "its", "and",
    "or", "but", "not", "no",
}

THRESHOLDS = {
    "exact_match": 1.0,
    "answer_coverage": 1.0,
    "keyword_f1": 0.6,
    "sequence_similarity": 0.7,
}


@dataclass
class Score:
    test_id: str
    exact_match: float
    answer_coverage: float
    keyword_f1: float
    sequence_similarity: float
    passed: bool
    reason: str


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(response: str, expected: str) -> float:
    return 1.0 if normalize(response) == normalize(expected) else 0.0


def keywords(text: str) -> set[str]:
    tokens = set(normalize(text).split())
    filtered = tokens - _STOP
    return filtered if filtered else tokens


def answer_coverage(response: str, expected: str) -> float:
    exp_kw = keywords(expected)
    res_kw = keywords(response)

    if not exp_kw:
        return 1.0

    overlap = len(res_kw & exp_kw)
    return overlap / len(exp_kw)


def keyword_f1(response: str, expected: str) -> float:
    exp_kw = keywords(expected)
    res_kw = keywords(response)

    if not exp_kw:
        return 1.0

    overlap = len(res_kw & exp_kw)
    precision = overlap / len(res_kw) if res_kw else 0.0
    recall = overlap / len(exp_kw)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_similarity(response: str, expected: str) -> float:
    return difflib.SequenceMatcher(
        None, normalize(response), normalize(expected)
    ).ratio()


def score(test_id: str, response: str, expected: str) -> Score:
    em = exact_match(response, expected)
    coverage = answer_coverage(response, expected)
    kf1 = keyword_f1(response, expected)
    ss = sequence_similarity(response, expected)

    if em >= THRESHOLDS["exact_match"]:
        passed, reason = True, "exact match"
    elif coverage >= THRESHOLDS["answer_coverage"]:
        passed, reason = True, f"answer coverage={coverage:.2f}"
    elif kf1 >= THRESHOLDS["keyword_f1"]:
        passed, reason = True, f"keyword F1={kf1:.2f}"
    elif ss >= THRESHOLDS["sequence_similarity"]:
        passed, reason = True, f"sequence similarity={ss:.2f}"
    else:
        passed = False
        reason = (
            "all below threshold - "
            f"exact={em:.2f}, coverage={coverage:.2f}, "
            f"kf1={kf1:.2f}, seq={ss:.2f}"
        )

    return Score(
        test_id=test_id,
        exact_match=em,
        answer_coverage=coverage,
        keyword_f1=kf1,
        sequence_similarity=ss,
        passed=passed,
        reason=reason,
    )
