"""
Scoring module for the LLM eval harness.

Four mechanisms — first three are stdlib, fourth requires sentence-transformers:

  1. exact_match        — normalized string equality (binary 1.0 / 0.0)
  2. keyword_f1         — precision/recall/F1 over non-stopword token sets
  3. sequence_similarity — difflib SequenceMatcher ratio
  4. semantic_sim       — cosine similarity over sentence embeddings (all-MiniLM-L6-v2)
                          falls back gracefully to None if sentence-transformers not installed

A test case passes when ANY mechanism meets its threshold:
  exact_match >= 1.0  OR  keyword_f1 >= 0.6  OR  sequence_similarity >= 0.7
  OR  semantic_sim >= 0.75

semantic_sim catches cases lexical methods miss — e.g. "two weeks annual leave"
vs "14 days annual leave": same meaning, zero keyword overlap.

Install semantic scoring: pip install -e ".[semantic]"
First call loads the model (~90 MB, ~400 ms); subsequent calls are fast.
"""

import re
import difflib
from dataclasses import dataclass
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    import sentence_transformers.util as _st_util
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model

# Stopwords excluded from keyword_f1 to avoid penalising phrasing differences
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "as", "into", "that", "this", "it", "its", "and",
    "or", "but", "not", "no",
}

THRESHOLDS = {
    "exact_match": 1.0,
    "keyword_f1": 0.6,
    "sequence_similarity": 0.7,
    "semantic_sim": 0.65,
}


@dataclass
class Score:
    test_id: str
    exact_match: float
    keyword_f1: float
    sequence_similarity: float
    semantic_sim: Optional[float]  # None when sentence-transformers not installed
    passed: bool
    reason: str


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(response: str, expected: str) -> float:
    return 1.0 if normalize(response) == normalize(expected) else 0.0


def keyword_f1(response: str, expected: str) -> float:
    def keywords(text: str) -> set[str]:
        tokens = set(normalize(text).split())
        filtered = tokens - _STOP
        return filtered if filtered else tokens  # fall back if only stopwords

    exp_kw = keywords(expected)
    res_kw = keywords(response)

    if not exp_kw:
        return 1.0  # nothing expected → trivially satisfied

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


def semantic_sim(response: str, expected: str) -> Optional[float]:
    """Cosine similarity between sentence embeddings. Returns None if sentence-transformers not installed."""
    if not _SEMANTIC_AVAILABLE:
        return None
    if not response.strip() or not expected.strip():
        return 0.0
    model = _get_semantic_model()
    embs = model.encode([response, expected], convert_to_tensor=True)
    return float(_st_util.cos_sim(embs[0], embs[1]))


def score(test_id: str, response: str, expected: str) -> Score:
    em = exact_match(response, expected)
    kf1 = keyword_f1(response, expected)
    ss = sequence_similarity(response, expected)
    sem = semantic_sim(response, expected)

    if em >= THRESHOLDS["exact_match"]:
        passed, reason = True, "exact match"
    elif kf1 >= THRESHOLDS["keyword_f1"]:
        passed, reason = True, f"keyword F1={kf1:.2f}"
    elif ss >= THRESHOLDS["sequence_similarity"]:
        passed, reason = True, f"sequence similarity={ss:.2f}"
    elif sem is not None and sem >= THRESHOLDS["semantic_sim"]:
        passed, reason = True, f"semantic similarity={sem:.2f}"
    else:
        passed = False
        scores_str = f"exact={em:.2f}, kf1={kf1:.2f}, seq={ss:.2f}"
        if sem is not None:
            scores_str += f", sem={sem:.2f}"
        reason = f"all below threshold — {scores_str}"

    return Score(
        test_id=test_id,
        exact_match=em,
        keyword_f1=kf1,
        sequence_similarity=ss,
        semantic_sim=sem,
        passed=passed,
        reason=reason,
    )
