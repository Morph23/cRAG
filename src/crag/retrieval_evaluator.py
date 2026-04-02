from __future__ import annotations

import re
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as cfg


class LLMRelevanceEvaluator:

    SYSTEM_PROMPT = (
        "You are a retrieval quality evaluator. "
        "Given a query and a document, output a single floating-point number "
        "in the range [-1, 1] indicating whether the document DIRECTLY ANSWERS the query. "
        "1.0 = the document contains a clear, direct answer to the query. "
        "-1.0 = the document does not answer the query (even if it is on the same topic). "
        "Being on the same topic is NOT enough — score high only if the document "
        "actually provides the information the query is asking for. "
        "Output ONLY the number, nothing else."
    )

    def __init__(self, model: str = cfg.GENERATOR_MODEL):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if "claude" in self.model.lower():
            import anthropic
            self._client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        else:
            import openai
            self._client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
        return self._client

    def _parse_score(self, text: str) -> float:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not matches:
            return 0.0
        return max(-1.0, min(1.0, float(matches[0])))

    def score(self, query: str, document: str) -> float:
        client = self._get_client()
        prompt = f"Query: {query}\n\nDocument: {document}"

        if "claude" in self.model.lower():
            resp = client.messages.create(
                model=self.model,
                max_tokens=16,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        else:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            text = resp.choices[0].message.content

        return self._parse_score(text)

    def score_batch(self, query: str, documents: List[str]) -> List[float]:
        return [self.score(query, doc) for doc in documents]


class RetrievalDecision:
    CORRECT   = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


def decide(
    scores: List[float],
    upper: float,
    lower: float,
) -> Tuple[str, List[float]]:
    if any(s >= upper for s in scores):
        return RetrievalDecision.CORRECT, scores
    if all(s <= lower for s in scores):
        return RetrievalDecision.INCORRECT, scores
    return RetrievalDecision.AMBIGUOUS, scores
