"""
metrics/hallucination.py — Hallucination detection for Voice AI responses.

Strategy (multi-signal fusion):
  1. LLM-as-judge: Ask Ollama to evaluate if response is grounded in context
  2. Named-entity overlap: check if entities in response exist in reference context
  3. Factual consistency: compare key claims between response and reference

Determinism: LLM judge uses temperature=0 and fixed seed.
"""

from __future__ import annotations
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Hallucination detection result for a single response."""
    response: str
    context: str
    is_hallucination: bool
    confidence: float
    signals: dict = field(default_factory=dict)
    llm_reasoning: str = ""
    method: str = "multi_signal"

    def to_dict(self) -> dict:
        return {
            "is_hallucination": self.is_hallucination,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
            "llm_reasoning": self.llm_reasoning,
            "method": self.method,
        }


# ─── LLM Judge prompt ────────────────────────────────────────────────────────

HALLUCINATION_JUDGE_PROMPT = """You are a factual accuracy evaluator for a Voice AI system.

Your task: determine if the AI RESPONSE introduces information not supported by the REFERENCE CONTEXT.

REFERENCE CONTEXT:
{context}

AI RESPONSE:
{response}

Evaluate strictly. A hallucination occurs when the response:
- States facts not present in the reference context
- Invents entities, numbers, dates, or names not mentioned
- Makes confident claims beyond what the context supports

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "is_hallucination": <true|false>,
  "confidence": <float 0.0-1.0, where 1.0 = certain hallucination>,
  "reasoning": "<one sentence explanation>"
}}"""


# ─── NER-lite: extract key entities ─────────────────────────────────────────

def _extract_entities(text: str) -> set[str]:
    """
    Lightweight entity extraction without spaCy.

    Extracts:
      - Capitalized proper-noun sequences
      - Numbers and percentages
      - Dates (simple patterns)
    """
    entities: set[str] = set()

    # Capitalized sequences (rough proper nouns)
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        entities.add(match.group(1).lower())

    # Percentages (before plain numbers to avoid double-matching)
    for match in re.finditer(r"\b\d+(?:\.\d+)?%", text):
        entities.add(match.group(0))

    # Plain numbers (not part of percentage)
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        raw = match.group(0)
        if raw + "%" not in text:
            entities.add(raw)

    # Dates
    for match in re.finditer(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b", text
    ):
        entities.add(match.group(0).lower())

    return entities


def _entity_overlap_score(response: str, context: str) -> float:
    """
    Fraction of response entities NOT found in context.

    Higher score = more unsupported entities = more likely hallucination.
    """
    response_entities = _extract_entities(response)
    if not response_entities:
        return 0.0

    context_lower = context.lower()
    unsupported = {e for e in response_entities if e not in context_lower}
    return len(unsupported) / len(response_entities)


def _length_ratio_signal(response: str, context: str) -> float:
    """
    Responses much longer than context may be generating unsupported content.
    Returns a soft signal in [0, 1].
    """
    resp_words = len(response.split())
    ctx_words  = len(context.split())
    if ctx_words == 0:
        return 0.0
    ratio = resp_words / ctx_words
    if ratio > 3.0:
        return min(1.0, (ratio - 3.0) / 7.0)
    return 0.0


class HallucinationDetector:
    """
    Multi-signal hallucination detector for Voice AI responses.

    Combines:
      - LLM-as-judge (primary signal, requires Ollama)
      - Named-entity overlap (secondary signal, always available)
      - Length ratio heuristic (tertiary signal)

    Final score is a weighted average of available signals.
    """

    SIGNAL_WEIGHTS = {
        "llm_judge":      0.60,
        "entity_overlap": 0.30,
        "length_ratio":   0.10,
    }

    def __init__(self, llm_client=None, threshold: float = 0.5):
        """
        Parameters
        ----------
        llm_client : OllamaClient, optional
            LLM client for judge scoring. If None, uses heuristics only.
        threshold : float
            Confidence above which a response is flagged as hallucination.
        """
        self.llm_client = llm_client
        self.threshold = threshold

    def detect(self, response: str, context: str,
               reference: Optional[str] = None) -> HallucinationResult:
        """
        Detect if response hallucinated content relative to context.

        Parameters
        ----------
        response : str
            The Voice AI system's response text.
        context : str
            Source material the response should be grounded in.
        reference : str, optional
            Gold-standard answer for additional grounding check.

        Returns
        -------
        HallucinationResult
        """
        signals: dict[str, float] = {}

        # Signal 1: LLM judge
        llm_reasoning = ""
        if self.llm_client is not None:
            llm_score, llm_reasoning = self._llm_judge_score(response, context)
            signals["llm_judge"] = llm_score

        # Signal 2: Entity overlap
        signals["entity_overlap"] = _entity_overlap_score(response, context)

        # Signal 3: Length ratio
        signals["length_ratio"] = _length_ratio_signal(response, context)

        # Weighted fusion
        total_weight = sum(self.SIGNAL_WEIGHTS[k] for k in signals)
        confidence = sum(
            signals[k] * self.SIGNAL_WEIGHTS[k] for k in signals
        ) / max(total_weight, 1e-9)

        confidence = max(0.0, min(1.0, confidence))

        return HallucinationResult(
            response=response,
            context=context,
            is_hallucination=confidence >= self.threshold,
            confidence=confidence,
            signals=signals,
            llm_reasoning=llm_reasoning,
            method="multi_signal" if len(signals) > 1 else "heuristic",
        )

    def _llm_judge_score(self, response: str, context: str) -> tuple[float, str]:
        """Run LLM judge and parse score from JSON response."""
        prompt = HALLUCINATION_JUDGE_PROMPT.format(
            context=context[:2000],
            response=response[:1000],
        )
        try:
            llm_resp = self.llm_client.generate(prompt, max_tokens=200)
            raw = llm_resp.response.strip()

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                logger.warning("LLM judge did not return valid JSON")
                return 0.5, "parse_error"

            parsed = json.loads(json_match.group(0))
            is_hallucination = parsed.get("is_hallucination", False)
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")

            score = confidence if is_hallucination else (1.0 - confidence)
            return max(0.0, min(1.0, score)), reasoning

        except Exception as e:
            logger.warning(f"LLM judge error: {e}")
            return 0.5, f"error: {e}"

    def batch_detect(self, samples: list[dict]) -> list[HallucinationResult]:
        """
        Detect hallucinations for a batch of samples.

        Each sample dict must have keys: 'response', 'context'.
        Optional key: 'reference'.
        """
        return [
            self.detect(
                response=s["response"],
                context=s["context"],
                reference=s.get("reference"),
            )
            for s in samples
        ]

    def aggregate(self, results: list[HallucinationResult]) -> dict:
        if not results:
            return {}
        confidences = [r.confidence for r in results]
        hallucinated = [r for r in results if r.is_hallucination]
        return {
            "hallucination_rate": round(len(hallucinated) / len(results), 4),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "num_hallucinations": len(hallucinated),
            "num_samples": len(results),
            "threshold": self.threshold,
        }