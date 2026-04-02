import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ThresholdConfig:
    upper: float
    lower: float


DATASET_THRESHOLDS: dict[str, ThresholdConfig] = {
    "popqa":         ThresholdConfig(upper=0.59,  lower=-0.99),
    "pubhealth":     ThresholdConfig(upper=0.50,  lower=-0.91),
    "arc_challenge": ThresholdConfig(upper=0.50,  lower=-0.91),
    "biography":     ThresholdConfig(upper=0.95,  lower=-0.91),
    "default":       ThresholdConfig(upper=0.50,  lower=-0.50),
}

KNOWLEDGE_STRIP_SCORE_THRESHOLD = -0.5
TOP_K_STRIPS = 5
MAX_RETRIEVED_DOCS = 5
WEB_SEARCH_RESULTS = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY:    Optional[str] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY:    Optional[str] = os.getenv("TAVILY_API_KEY")
SERP_API_KEY:      Optional[str] = os.getenv("SERP_API_KEY")

GENERATOR_MODEL     = os.getenv("GENERATOR_MODEL",     "gpt-3.5-turbo")
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "gpt-3.5-turbo")
