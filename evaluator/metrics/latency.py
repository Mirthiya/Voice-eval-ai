"""
metrics/latency.py — End-to-end and component latency measurement.

Measures:
  - Time-to-first-token (TTFT)
  - Total response latency
  - Audio transcription latency
  - LLM inference latency
  - Real-Time Factor (RTF) = processing_time / audio_duration

Determinism: timing is measurement-only and does not affect outputs.
"""

from __future__ import annotations
import time
import statistics
import contextlib
from dataclasses import dataclass, field
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class LatencyRecord:
    """Single latency measurement with component breakdown."""
    label: str
    total_ms: float
    transcription_ms: float = 0.0
    llm_ms: float = 0.0
    scoring_ms: float = 0.0
    audio_duration_s: float = 0.0

    @property
    def real_time_factor(self) -> float:
        """RTF = processing_time / audio_duration. <1.0 = faster than real-time."""
        if self.audio_duration_s <= 0:
            return 0.0
        return round((self.total_ms / 1000) / self.audio_duration_s, 4)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total_ms": round(self.total_ms, 2),
            "transcription_ms": round(self.transcription_ms, 2),
            "llm_ms": round(self.llm_ms, 2),
            "scoring_ms": round(self.scoring_ms, 2),
            "real_time_factor": self.real_time_factor,
            "audio_duration_s": round(self.audio_duration_s, 3),
        }


@dataclass
class LatencyStats:
    """Aggregate latency statistics across all evaluation samples."""
    records: list[LatencyRecord] = field(default_factory=list)
    percentiles: list[int] = field(default_factory=lambda: [50, 90, 95, 99])

    def add(self, record: LatencyRecord) -> None:
        self.records.append(record)

    def compute(self) -> dict:
        """Compute descriptive statistics over all recorded latencies."""
        if not self.records:
            return {}

        totals = [r.total_ms for r in self.records]
        transcriptions = [r.transcription_ms for r in self.records]
        llm_latencies = [r.llm_ms for r in self.records]
        rtfs = [r.real_time_factor for r in self.records if r.real_time_factor > 0]

        def _pct(values: list[float], p: int) -> float:
            if not values:
                return 0.0
            sorted_v = sorted(values)
            k = (len(sorted_v) - 1) * p / 100
            lo, hi = int(k), min(int(k) + 1, len(sorted_v) - 1)
            return round(sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (k - lo), 2)

        result: dict = {
            "total_ms": {
                "mean": round(statistics.mean(totals), 2),
                "median": round(statistics.median(totals), 2),
                "stdev": round(statistics.stdev(totals) if len(totals) > 1 else 0.0, 2),
                "min": round(min(totals), 2),
                "max": round(max(totals), 2),
            },
            "transcription_ms": {
                "mean": round(statistics.mean(transcriptions), 2),
                "median": round(statistics.median(transcriptions), 2),
            },
            "llm_ms": {
                "mean": round(statistics.mean(llm_latencies), 2),
                "median": round(statistics.median(llm_latencies), 2),
            },
            "real_time_factor": {
                "mean": round(statistics.mean(rtfs), 4) if rtfs else 0.0,
                "max": round(max(rtfs), 4) if rtfs else 0.0,
            },
            "percentiles_ms": {
                f"p{p}": _pct(totals, p) for p in self.percentiles
            },
            "num_samples": len(self.records),
        }
        return result


@contextlib.contextmanager
def timer(label: str = ""):
    """Context manager for inline latency measurement."""
    t0 = time.perf_counter()
    result = {"ms": 0.0, "label": label}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - t0) * 1000


def timed(fn: Callable[P, T]) -> Callable[P, tuple[T, float]]:
    """
    Decorator that returns (result, elapsed_ms) for any function call.

    Usage:
        result, ms = timed(my_function)(arg1, arg2)
    """
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, float]:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return result, elapsed_ms
    wrapper.__name__ = fn.__name__
    return wrapper  # type: ignore