"""
metrics/semantic_similarity.py — Semantic similarity scoring.

Primary method: sentence-transformers cosine similarity (fast, local, no API calls)
Fallback method: TF-IDF + cosine similarity (pure Python, zero dependencies)

Determinism: embeddings are deterministic for fixed model weights and input.
"""

from __future__ import annotations
import math
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Semantic similarity between two texts."""
    text_a: str
    text_b: str
    score: float
    method: str
    is_similar: bool
    threshold: float

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "method": self.method,
            "is_similar": self.is_similar,
            "threshold": self.threshold,
        }


# ─── TF-IDF fallback (pure Python) ───────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _tfidf_cosine(a: str, b: str) -> float:
    """Compute TF-IDF weighted cosine similarity between two strings."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    vocab = list(set(tokens_a) | set(tokens_b))

    def tf(tokens: list[str]) -> dict[str, float]:
        counts: dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = len(tokens)
        return {k: v / total for k, v in counts.items()}

    def idf(term: str) -> float:
        doc_count = sum(1 for tokens in [tokens_a, tokens_b] if term in tokens)
        return math.log((1 + 2) / (1 + doc_count)) + 1

    tf_a = tf(tokens_a)
    tf_b = tf(tokens_b)

    vec_a = [tf_a.get(t, 0.0) * idf(t) for t in vocab]
    vec_b = [tf_b.get(t, 0.0) * idf(t) for t in vocab]

    dot   = sum(x * y for x, y in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(y * y for y in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Sentence Transformer embeddings ────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticSimilarityCalculator:
    """
    Compute semantic similarity between two texts.

    Tries sentence-transformers first; falls back to TF-IDF cosine.
    Both are deterministic for fixed inputs.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 threshold: float = 0.75):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._method = "tfidf"
        self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._method = "sentence_transformer"
            logger.info("sentence-transformers loaded successfully")
        except ImportError:
            logger.info("sentence-transformers not installed — using TF-IDF fallback")
        except Exception as e:
            logger.warning(f"sentence-transformers failed to load ({e}) — using TF-IDF")

    def _embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def compute(self, text_a: str, text_b: str) -> SimilarityResult:
        """
        Compute semantic similarity between two texts.

        Parameters
        ----------
        text_a, text_b : str
            Texts to compare.

        Returns
        -------
        SimilarityResult
        """
        if text_a.strip().lower() == text_b.strip().lower():
            return SimilarityResult(
                text_a=text_a, text_b=text_b,
                score=1.0, method="exact",
                is_similar=True, threshold=self.threshold,
            )

        if self._model is not None:
            emb_a = self._embed(text_a)
            emb_b = self._embed(text_b)
            score = _cosine(emb_a, emb_b)
            method = "sentence_transformer"
        else:
            score = _tfidf_cosine(text_a, text_b)
            method = "tfidf"

        score = max(0.0, min(1.0, score))

        return SimilarityResult(
            text_a=text_a,
            text_b=text_b,
            score=score,
            method=method,
            is_similar=score >= self.threshold,
            threshold=self.threshold,
        )

    def batch_compute(self, pairs: list[tuple[str, str]]) -> list[SimilarityResult]:
        """Compute similarity for a list of text pairs."""
        if self._model is not None and len(pairs) > 1:
            return self._batch_sentence_transformer(pairs)
        return [self.compute(a, b) for a, b in pairs]

    def _batch_sentence_transformer(self, pairs: list[tuple[str, str]]) -> list[SimilarityResult]:
        """Efficient batched embedding computation."""
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        all_texts = texts_a + texts_b

        embeddings = self._model.encode(
            all_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        n = len(pairs)
        embs_a = embeddings[:n]
        embs_b = embeddings[n:]

        results = []
        for i, (a, b) in enumerate(pairs):
            score = float(_cosine(embs_a[i].tolist(), embs_b[i].tolist()))
            score = max(0.0, min(1.0, score))
            results.append(SimilarityResult(
                text_a=a, text_b=b,
                score=score, method="sentence_transformer",
                is_similar=score >= self.threshold,
                threshold=self.threshold,
            ))
        return results

    def aggregate(self, results: list[SimilarityResult]) -> dict:
        scores = [r.score for r in results]
        if not scores:
            return {}
        return {
            "mean_similarity": round(sum(scores) / len(scores), 4),
            "min_similarity": round(min(scores), 4),
            "max_similarity": round(max(scores), 4),
            "similar_rate": round(sum(1 for r in results if r.is_similar) / len(results), 4),
            "threshold": self.threshold,
            "method": results[0].method if results else "unknown",
            "num_samples": len(results),
        }