
from __future__ import annotations

import re
from typing import List, Tuple

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as cfg

AnyEvaluator = "LLMRelevanceEvaluator"  # noqa: F821


def _split_into_strips(text: str, min_sentences: int = 1, max_sentences: int = 2) -> List[str]:
    text = text.strip()
    if not text:
        return []

    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return [text]

    strips: List[str] = []
    step = max(1, max_sentences - 1)
    for i in range(0, len(sentences), step):
        window = sentences[i: i + max_sentences]
        strips.append(" ".join(window))

    return strips


def _score_strips(
    query: str,
    strips: List[str],
    evaluator: AnyEvaluator,
) -> List[Tuple[str, float]]:
    if hasattr(evaluator, "score_batch"):
        scores = evaluator.score_batch(query, strips)
    else:
        scores = [evaluator.score(query, s) for s in strips]
    return list(zip(strips, scores))


def refine_documents(
    query: str,
    documents: List[str],
    evaluator: AnyEvaluator,
    score_threshold: float = cfg.KNOWLEDGE_STRIP_SCORE_THRESHOLD,
    top_k: int = cfg.TOP_K_STRIPS,
) -> str:
    if not documents:
        return ""

    all_scored: List[Tuple[str, float, int]] = []
    global_idx = 0
    for doc in documents:
        strips = _split_into_strips(doc)
        scored = _score_strips(query, strips, evaluator)
        for strip, score in scored:
            all_scored.append((strip, score, global_idx))
            global_idx += 1

    if not all_scored:
        return ""

    surviving = [(s, sc, idx) for s, sc, idx in all_scored if sc >= score_threshold]

    if not surviving:
        surviving = [max(all_scored, key=lambda x: x[1])]

    top = sorted(surviving, key=lambda x: x[1], reverse=True)[:top_k]
    top_ordered = sorted(top, key=lambda x: x[2])

    return " ".join(s for s, _, _ in top_ordered)


def refine_single_document(
    query: str,
    document: str,
    evaluator: AnyEvaluator,
    score_threshold: float = cfg.KNOWLEDGE_STRIP_SCORE_THRESHOLD,
    top_k: int = cfg.TOP_K_STRIPS,
) -> str:
    return refine_documents(query, [document], evaluator, score_threshold, top_k)


def get_strip_scores(
    query: str,
    documents: List[str],
    evaluator: AnyEvaluator,
) -> List[Tuple[str, float]]:
    result: List[Tuple[str, float]] = []
    for doc in documents:
        strips = _split_into_strips(doc)
        scored = _score_strips(query, strips, evaluator)
        result.extend(scored)
    return result
