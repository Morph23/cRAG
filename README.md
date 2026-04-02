# Corrective RAG (CRAG)

Implementation of the paper [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (Yan et al., 2024).

The core idea is that standard RAG blindly trusts whatever documents it retrieves, even if they're irrelevant. CRAG adds a lightweight evaluator that scores each retrieved document and decides what to do — use it, discard it, or go search the web.

---

## How it works

Every query goes through three stages:

**1. Retrieval** — fetch top-k documents from a local FAISS index using dense embeddings.

**2. Evaluation** — score each document against the query on a scale of [-1, 1]. Based on the scores, one of three actions is taken:
- `CORRECT` — at least one doc is relevant → refine and use it
- `INCORRECT` — nothing is relevant → discard everything, run a web search
- `AMBIGUOUS` — mixed signals → use both retrieved docs and web results

**3. Generation** — pass the refined context to an LLM and generate the final answer.

The knowledge refinement step (decompose-then-recompose) splits documents into 1-2 sentence strips, scores each strip individually, filters out the weak ones, and concatenates the survivors. This prevents noisy content from polluting the context.

---

## Usage

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...        # recommended for web search
```

To use Claude instead of GPT, set:
```
ANTHROPIC_API_KEY=...
GENERATOR_MODEL=claude-sonnet-4-6
```

**Run the demo:**
```bash
python example.py
```

This runs 4 queries against a small in-memory corpus. The first two should resolve from the corpus (`CORRECT`), the last two will trigger web search (`INCORRECT` or 'AMBIGOUS').

**Single query:**
```bash
python example.py --query "What is the capital of France?"
```

**Inspect knowledge refinement strip scores:**
```bash
python example.py --inspect
```

**Use in your own code:**
```python
from src.crag import CRAGPipeline, LLMRelevanceEvaluator, FAISSRetriever
import config as cfg

retriever = FAISSRetriever(embedding_model=cfg.EMBEDDING_MODEL)
retriever.add_documents(["your documents here", "..."])

pipeline = CRAGPipeline(
    retriever=retriever,
    evaluator=LLMRelevanceEvaluator(model=cfg.GENERATOR_MODEL),
    threshold_config=cfg.DATASET_THRESHOLDS["default"],
    generator_model=cfg.GENERATOR_MODEL,
)

result = pipeline.run("your question")
print(result.action)   # correct / incorrect / ambiguous
print(result.answer)
```

---

## Configuration

Thresholds for the evaluator decision boundaries are in `config.py`. The paper reports dataset-specific values (Table 2):

| Dataset | Upper | Lower |
|---|---|---|
| PopQA | 0.59 | -0.99 |
| PubHealth / ARC | 0.50 | -0.91 |
| Biography | 0.95 | -0.91 |
| Default | 0.50 | -0.50 |

---

## Note

The paper uses a fine-tuned T5-large model as the retrieval evaluator. This implementation uses an LLM-based evaluator instead, which works well enough for demos but won't match the paper's benchmark numbers.

Web search priority: Tavily → SerpAPI → DuckDuckGo. Only Tavily is needed for reliable results.
