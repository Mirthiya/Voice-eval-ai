"""
Voice AI Evaluation Framework
==============================
A modular, deterministic pipeline for evaluating Voice AI systems.

Metrics: WER, Latency, Semantic Similarity, Hallucination Rate
Integrations: Whisper (transcription), Ollama (LLM scoring)
"""

__version__ = "1.0.0"
__author__ = "Voice AI Eval Framework"

from .pipeline import EvaluationPipeline
from .config import EvalConfig

__all__ = ["EvaluationPipeline", "EvalConfig"]
