
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as cfg


@dataclass
class Document:
    content:  str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        snippet = self.content[:80].replace("\n", " ")
        return f"Document(content='{snippet}...', metadata={self.metadata})"


class FAISSRetriever:

    def __init__(
        self,
        embedding_model: str = cfg.EMBEDDING_MODEL,
        device:          str | None = None,
    ):
        from sentence_transformers import SentenceTransformer
        self._device   = device or ("cuda" if self._cuda_available() else "cpu")
        self._encoder  = SentenceTransformer(embedding_model, device=self._device)
        self._dim      = self._encoder.get_sentence_embedding_dimension()
        self._index    = None
        self._docs: List[Document] = []

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _build_index(self, vectors: np.ndarray) -> None:
        import faiss
        index = faiss.IndexFlatIP(self._dim)  # inner-product (cosine with normalised vecs)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        self._index = index

    def add_documents(self, documents: List[str | Document]) -> None:
        import faiss

        new_docs: List[Document] = []
        for d in documents:
            if isinstance(d, str):
                new_docs.append(Document(content=d))
            else:
                new_docs.append(d)

        texts   = [d.content for d in new_docs]
        vectors = self._encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = vectors.astype(np.float32)

        if self._index is None:
            self._docs.extend(new_docs)
            self._build_index(vectors)
        else:
            self._docs.extend(new_docs)
            faiss.normalize_L2(vectors)
            self._index.add(vectors)

    def retrieve(
        self,
        query:         str,
        k:             int = cfg.MAX_RETRIEVED_DOCS,
        return_scores: bool = True,
    ) -> List[Tuple[Document, float]] | List[Document]:
        if self._index is None or len(self._docs) == 0:
            return []

        import faiss
        q_vec = self._encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        k_actual = min(k, len(self._docs))
        sims, idxs = self._index.search(q_vec, k_actual)

        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx == -1:
                continue
            doc = self._docs[idx]
            if return_scores:
                results.append((doc, float(sim)))
            else:
                results.append(doc)

        return results

    def save(self, directory: str) -> None:
        import faiss
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "docs.pkl"), "wb") as f:
            pickle.dump(self._docs, f)

    @classmethod
    def load(
        cls,
        directory:       str,
        embedding_model: str = cfg.EMBEDDING_MODEL,
        device:          str | None = None,
    ) -> "FAISSRetriever":
        import faiss
        obj = cls(embedding_model=embedding_model, device=device)
        obj._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "docs.pkl"), "rb") as f:
            obj._docs = pickle.load(f)
        return obj

    def __len__(self) -> int:
        return len(self._docs)
