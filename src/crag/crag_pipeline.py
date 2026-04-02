
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as cfg
from src.crag.retriever            import FAISSRetriever, Document
from src.crag.retrieval_evaluator  import (
    LLMRelevanceEvaluator,
    RetrievalDecision, decide,
)
from src.crag.knowledge_refinement import refine_documents, get_strip_scores
from src.crag.web_search           import web_search

logger = logging.getLogger(__name__)


@dataclass
class CRAGResult:
    query:           str
    answer:          str
    action:          str
    retrieved_docs:  List[Document] = field(default_factory=list)
    doc_scores:      List[float]   = field(default_factory=list)
    web_passages:    List[str]     = field(default_factory=list)
    refined_context: str           = ""
    metadata:        Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CRAGResult(\n"
            f"  query='{self.query}',\n"
            f"  action='{self.action}',\n"
            f"  answer='{self.answer[:120]}...'\n"
            f")"
        )


_GENERATOR_SYSTEM = (
    "You are a knowledgeable assistant. Use only the provided context to answer "
    "the question. If the context does not contain enough information to answer, "
    "say so clearly. Be concise and accurate."
)


def _call_llm(
    query:   str,
    context: str,
    model:   str = cfg.GENERATOR_MODEL,
) -> str:
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    if "claude" in model.lower():
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_GENERATOR_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()
    else:
        import openai
        client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _GENERATOR_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


class CRAGPipeline:

    def __init__(
        self,
        retriever:        FAISSRetriever,
        evaluator:        LLMRelevanceEvaluator | None = None,
        threshold_config: cfg.ThresholdConfig | None = None,
        generator_model:  str = cfg.GENERATOR_MODEL,
        top_k_docs:       int = cfg.MAX_RETRIEVED_DOCS,
        verbose:          bool = False,
    ):
        self.retriever     = retriever
        self.evaluator     = evaluator or LLMRelevanceEvaluator()
        self.thresholds    = threshold_config or cfg.DATASET_THRESHOLDS["default"]
        self.gen_model     = generator_model
        self.top_k         = top_k_docs
        self.verbose       = verbose

        if verbose:
            logging.basicConfig(level=logging.INFO)

    def _retrieve(self, query: str) -> tuple[List[Document], List[float]]:
        results = self.retriever.retrieve(query, k=self.top_k, return_scores=True)
        if not results:
            return [], []
        docs, sims = zip(*results)
        return list(docs), list(sims)

    def _evaluate(self, query: str, docs: List[Document]) -> tuple[str, List[float]]:
        if not docs:
            logger.info("No docs retrieved → INCORRECT")
            return RetrievalDecision.INCORRECT, []

        contents = [d.content for d in docs]
        scores   = self.evaluator.score_batch(query, contents) \
                   if hasattr(self.evaluator, "score_batch") \
                   else [self.evaluator.score(query, c) for c in contents]

        action, scores = decide(scores, self.thresholds.upper, self.thresholds.lower)
        logger.info("Evaluator scores: %s  →  action: %s", scores, action)
        return action, scores

    def _get_web_context(self, query: str) -> tuple[List[str], str]:
        passages = web_search(query, num_results=cfg.WEB_SEARCH_RESULTS, rewrite=True)
        if not passages:
            return [], ""
        truncated = [p[:1500] for p in passages]
        refined  = refine_documents(query, truncated, self.evaluator)
        return passages, refined

    def _refine_retrieved(self, query: str, docs: List[Document]) -> str:
        return refine_documents(query, [d.content for d in docs], self.evaluator)

    def _generate(self, query: str, context: str) -> str:
        if not context.strip():
            context = "No relevant context was found."
        return _call_llm(query, context, model=self.gen_model)

    def run(self, query: str) -> CRAGResult:
        docs, sims          = self._retrieve(query)
        action, eval_scores = self._evaluate(query, docs)

        web_passages: List[str] = []
        refined_context         = ""

        if action == RetrievalDecision.CORRECT:
            refined_context = self._refine_retrieved(query, docs)

        elif action == RetrievalDecision.INCORRECT:
            web_passages, refined_context = self._get_web_context(query)

        else:  # AMBIGUOUS
            retrieved_refined = self._refine_retrieved(query, docs)
            web_passages, web_refined = self._get_web_context(query)
            refined_context = (retrieved_refined + " " + web_refined).strip()

        answer = self._generate(query, refined_context)

        return CRAGResult(
            query           = query,
            answer          = answer,
            action          = action,
            retrieved_docs  = docs,
            doc_scores      = eval_scores,
            web_passages    = web_passages,
            refined_context = refined_context,
        )

    def run_batch(self, queries: List[str]) -> List[CRAGResult]:
        return [self.run(q) for q in queries]

    @classmethod
    def from_texts(
        cls,
        texts:            List[str],
        evaluator:        LLMRelevanceEvaluator | None = None,
        threshold_config: cfg.ThresholdConfig | None = None,
        generator_model:  str = cfg.GENERATOR_MODEL,
        embedding_model:  str = cfg.EMBEDDING_MODEL,
        verbose:          bool = False,
    ) -> "CRAGPipeline":
        retriever = FAISSRetriever(embedding_model=embedding_model)
        retriever.add_documents(texts)
        return cls(
            retriever        = retriever,
            evaluator        = evaluator,
            threshold_config = threshold_config,
            generator_model  = generator_model,
            verbose          = verbose,
        )
