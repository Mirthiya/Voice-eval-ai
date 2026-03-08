"""
config.py — Evaluation configuration with determinism guarantees.

All randomness is seeded. All model parameters are pinned.
This ensures identical results across evaluation runs.
"""

from __future__ import annotations
import os
import random
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Global determinism seed ─────────────────────────────────────────────────
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Pin all sources of randomness for reproducible evaluation runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed()


# ─── Configuration dataclass ─────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """
    Central configuration for the evaluation pipeline.

    All fields have defaults so the pipeline works out-of-the-box.
    Override only what you need — every run is fully reproducible.
    """

    # --- Determinism ---
    seed: int = GLOBAL_SEED

    # --- Whisper transcription ---
    whisper_model: str = "base"          # tiny | base | small | medium | large
    whisper_language: Optional[str] = None  # None = auto-detect
    whisper_fp16: bool = False           # False ensures CPU determinism
    whisper_beam_size: int = 5           # Fixed beam for determinism

    # --- Ollama / LLM ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"     # Any model pulled in Ollama
    ollama_temperature: float = 0.0      # 0.0 = deterministic greedy decoding
    ollama_seed: int = GLOBAL_SEED
    ollama_timeout: int = 60             # seconds

    # --- Semantic similarity ---
    similarity_model: str = "all-MiniLM-L6-v2"   # sentence-transformers model
    similarity_threshold: float = 0.75

    # --- Hallucination detection ---
    hallucination_model: str = ollama_model       # reuse same LLM
    hallucination_threshold: float = 0.5          # 0-1 score; above = hallucination

    # --- Latency ---
    latency_warmup_runs: int = 1        # discard first N runs (JIT warmup)
    latency_percentiles: list[int] = field(default_factory=lambda: [50, 90, 95, 99])

    # --- Reporting ---
    report_dir: str = "reports"
    report_indent: int = 2

    # --- Misc ---
    max_workers: int = 1                 # Serial by default for determinism
    audio_sample_rate: int = 16000      # Hz — Whisper standard

    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the config for traceability."""
        import json
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return self.__dict__.copy()