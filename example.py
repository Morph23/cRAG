
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src.crag import CRAGPipeline, LLMRelevanceEvaluator, FAISSRetriever
import config as cfg


CORPUS = [
    "The French Revolution began in 1789 with the storming of the Bastille on July 14. "
    "It was a period of radical political and societal change in France that overthrew the "
    "monarchy, established a republic, and culminated in Napoleon Bonaparte's rise to power.",

    "Marie Curie was a pioneering physicist and chemist who conducted research on radioactivity. "
    "She was the first woman to win a Nobel Prize, the only person to win Nobel Prizes in two "
    "different sciences (Physics 1903, Chemistry 1911), and the first woman to receive a PhD "
    "in France.",

    "The Treaty of Versailles was signed on June 28, 1919, officially ending World War I. "
    "It imposed heavy reparations on Germany, stripped it of territories, and limited its "
    "military forces, widely considered a contributing factor to World War II.",

    "Photosynthesis is the process by which green plants and some other organisms use sunlight, "
    "water, and carbon dioxide to produce oxygen and energy in the form of sugar. The chemical "
    "equation is 6CO2 + 6H2O + light → C6H12O6 + 6O2.",

    "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in living "
    "organisms. Its double-helix structure was discovered by James Watson and Francis Crick "
    "in 1953, building on X-ray crystallography work by Rosalind Franklin.",

    "Black holes are regions of spacetime where gravity is so strong that nothing — not even "
    "light or other electromagnetic waves — can escape once past the event horizon. The "
    "first image of a black hole was captured by the Event Horizon Telescope in 2019.",

    "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' "
    "by Vaswani et al., revolutionized natural language processing by replacing recurrent "
    "neural networks with self-attention mechanisms, enabling parallelisation and better "
    "long-range dependency modeling.",

    "Python was created by Guido van Rossum and first released in 1991. It is an interpreted, "
    "high-level, general-purpose programming language known for its readable syntax. It is the "
    "most popular language for machine learning and data science.",
]


def build_pipeline(verbose: bool = True) -> CRAGPipeline:
    evaluator = LLMRelevanceEvaluator(model=cfg.GENERATOR_MODEL)

    retriever = FAISSRetriever(embedding_model=cfg.EMBEDDING_MODEL)
    retriever.add_documents(CORPUS)

    pipeline = CRAGPipeline(
        retriever        = retriever,
        evaluator        = evaluator,
        threshold_config = cfg.DATASET_THRESHOLDS["default"],
        generator_model  = cfg.GENERATOR_MODEL,
        verbose          = verbose,
    )
    print(f"Pipeline ready — {len(retriever)} documents indexed.\n")
    return pipeline


QUERIES = [
    "When did the French Revolution begin and what was its significance?",
    "Who discovered the double-helix structure of DNA?",
    "What were the most recent developments in large language model research?",
    "What is the current population of Tokyo?",
]


def run_demo() -> None:
    pipeline = build_pipeline(verbose=False)

    for i, query in enumerate(QUERIES, 1):
        print(f"{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}")

        result = pipeline.run(query)

        print(f"Action taken  : {result.action.upper()}")
        print(f"Doc scores    : {[f'{s:.3f}' for s in result.doc_scores]}")
        if result.retrieved_docs:
            print(f"Top doc (snippet): {result.retrieved_docs[0].content[:120]}...")
        if result.web_passages:
            print(f"Web passages  : {len(result.web_passages)} fetched")
        print(f"\nAnswer:\n{result.answer}")
        print()


def inspect_knowledge_refinement() -> None:
    from src.crag.knowledge_refinement import get_strip_scores

    evaluator = LLMRelevanceEvaluator(model=cfg.GENERATOR_MODEL)

    query = "What is the transformer architecture?"
    doc   = (
        "The transformer architecture, introduced in 2017 paper 'Attention Is All You Need' "
        "by Vaswani et al., revolutionized natural language processing. "
        "It replaced recurrent neural networks with self-attention mechanisms. "
        "This enables parallelisation and better long-range dependency modeling. "
        "Transformers have become the basis for large language models like GPT and BERT."
    )

    scored = get_strip_scores(query, [doc], evaluator)
    print("\nKnowledge Strip Scores:")
    print(f"{'Strip':<80} {'Score':>7}")
    print("-" * 90)
    for strip, score in scored:
        print(f"{strip[:78]:<80} {score:>7.3f}")


def evaluate_on_popqa(pipeline: CRAGPipeline, data_path: str, max_samples: int = 50) -> float:
    import json

    correct = 0
    total   = 0

    with open(data_path) as f:
        for line in f:
            if total >= max_samples:
                break
            item = json.loads(line.strip())
            question = item["question"]
            answers  = [a.lower() for a in item.get("possible_answers", [])]

            result = pipeline.run(question)
            pred   = result.answer.lower()

            if any(ans in pred for ans in answers):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"PopQA Accuracy ({total} samples): {acc:.3f}")
    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CRAG demo")
    parser.add_argument("--inspect", action="store_true",
                        help="Show knowledge refinement strip scores instead of full demo")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single custom query")
    args = parser.parse_args()

    if args.inspect:
        inspect_knowledge_refinement()
    elif args.query:
        pipeline = build_pipeline(verbose=True)
        result   = pipeline.run(args.query)
        print(f"\nAction : {result.action.upper()}")
        print(f"Answer : {result.answer}")
    else:
        run_demo()
