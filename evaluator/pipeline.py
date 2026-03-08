"""
pipeline.py — Core evaluation pipeline orchestrator.

Coordinates:
  1. Audio transcription (Whisper)
  2. LLM response generation (Ollama)
  3. Metric computation (WER, Latency, Semantic Similarity, Hallucination)
  4. Report generation (JSON)

Determinism strategy:
  - All components seeded with GLOBAL_SEED
  - Evaluation order is fixed (sorted by sample ID)
  - No parallelism by default (max_workers=1)
  - Every run with the same config + data produces bit-identical metrics
"""

from __future__ import annotations
import time
import uuid
import logging
import datetime
from dataclasses import dataclass, field
from typing import Optional

from .config import EvalConfig, set_global_seed
from .transcription import WhisperTranscriber, TranscriptionResult
from .llm import OllamaClient
from .metrics import (
    WERCalculator, WERResult,
    LatencyRecord, LatencyStats,
    SemanticSimilarityCalculator, SimilarityResult,
    HallucinationDetector, HallucinationResult,
    timer,
)
from .reporter import ReportWriter

logger = logging.getLogger(__name__)


# ─── Sample schema ────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """A single evaluation sample."""
    sample_id: str
    audio_path: str
    reference_transcript: str
    reference_response: str
    context: str = ""
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.audio_path:
            raise ValueError(f"[{self.sample_id}] audio_path is required")
        if not self.reference_transcript:
            raise ValueError(f"[{self.sample_id}] reference_transcript is required")


@dataclass
class SampleResult:
    """All metrics computed for a single EvalSample."""
    sample_id: str
    audio_path: str

    transcription: Optional[TranscriptionResult] = None
    llm_response: Optional[str] = None
    llm_latency_ms: float = 0.0

    wer: Optional[WERResult] = None
    similarity: Optional[SimilarityResult] = None
    hallucination: Optional[HallucinationResult] = None
    latency: Optional[LatencyRecord] = None

    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "audio_path": self.audio_path,
            "transcription": self.transcription.to_dict() if self.transcription else None,
            "llm_response": self.llm_response,
            "metrics": {
                "wer": self.wer.to_dict() if self.wer else None,
                "semantic_similarity": self.similarity.to_dict() if self.similarity else None,
                "hallucination": self.hallucination.to_dict() if self.hallucination else None,
                "latency": self.latency.to_dict() if self.latency else None,
            },
            "errors": self.errors,
        }


# ─── Pipeline ────────────────────────────────────────────────────────────────

class EvaluationPipeline:
    """
    End-to-end Voice AI evaluation pipeline.

    Usage
    -----
    >>> from evaluator import EvaluationPipeline, EvalConfig
    >>> pipeline = EvaluationPipeline(EvalConfig())
    >>> report = pipeline.run(samples)
    >>> pipeline.save_report(report, "reports/run_001.json")
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()

        set_global_seed(self.config.seed)

        logger.info(f"Initialising evaluation pipeline (seed={self.config.seed})")

        self.transcriber = WhisperTranscriber(
            model_name=self.config.whisper_model,
            language=self.config.whisper_language,
            fp16=self.config.whisper_fp16,
            beam_size=self.config.whisper_beam_size,
        )

        self.llm = OllamaClient(
            host=self.config.ollama_host,
            model=self.config.ollama_model,
            temperature=self.config.ollama_temperature,
            seed=self.config.ollama_seed,
            timeout=self.config.ollama_timeout,
        )

        self.wer_calc = WERCalculator(normalize=True)

        self.similarity_calc = SemanticSimilarityCalculator(
            model_name=self.config.similarity_model,
            threshold=self.config.similarity_threshold,
        )

        self.hallucination_detector = HallucinationDetector(
            llm_client=self.llm,
            threshold=self.config.hallucination_threshold,
        )

        self.latency_stats = LatencyStats(
            percentiles=self.config.latency_percentiles,
        )

        self.reporter = ReportWriter(
            report_dir=self.config.report_dir,
            indent=self.config.report_indent,
        )

        logger.info("Pipeline initialised")

    def run(self, samples: list[EvalSample],
            run_id: Optional[str] = None) -> dict:
        """
        Run full evaluation over a list of EvalSamples.

        Parameters
        ----------
        samples : list[EvalSample]
            Evaluation dataset. Order is sorted by sample_id for determinism.
        run_id : str, optional
            Unique identifier for this run. Auto-generated if not provided.

        Returns
        -------
        dict
            Complete evaluation report (JSON-serialisable).
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        started_at = datetime.datetime.utcnow().isoformat() + "Z"
        wall_start = time.perf_counter()

        logger.info(f"Starting evaluation run {run_id} — {len(samples)} samples")

        samples_sorted = sorted(samples, key=lambda s: s.sample_id)

        for sample in samples_sorted:
            try:
                sample.validate()
            except ValueError as e:
                logger.error(f"Invalid sample: {e}")
                raise

        results: list[SampleResult] = []
        for i, sample in enumerate(samples_sorted):
            logger.info(f"  [{i+1}/{len(samples_sorted)}] Evaluating: {sample.sample_id}")
            result = self._evaluate_sample(sample)
            results.append(result)

        wall_elapsed_s = time.perf_counter() - wall_start
        aggregate = self._aggregate(results)

        report = {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "wall_time_seconds": round(wall_elapsed_s, 3),
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "num_samples": len(samples_sorted),
            "aggregate_metrics": aggregate,
            "sample_results": [r.to_dict() for r in results],
        }

        logger.info(f"Run {run_id} complete in {wall_elapsed_s:.2f}s")
        return report

    def save_report(self, report: dict, path: Optional[str] = None) -> str:
        """Save report to JSON file. Returns the file path."""
        return self.reporter.save(report, path)

    def _evaluate_sample(self, sample: EvalSample) -> SampleResult:
        """Evaluate a single sample through the full pipeline."""
        result = SampleResult(
            sample_id=sample.sample_id,
            audio_path=sample.audio_path,
        )
        t_transcription_ms = 0.0
        t_llm_ms = 0.0

        # ── Step 1: Transcription ────────────────────────────────────────────
        try:
            with timer("transcription") as t:
                transcription = self.transcriber.transcribe(sample.audio_path)
            t_transcription_ms = t["ms"]
            result.transcription = transcription
        except Exception as e:
            logger.error(f"  Transcription failed for {sample.sample_id}: {e}")
            result.errors.append(f"transcription: {e}")
            transcription = None

        # ── Step 2: LLM response generation ─────────────────────────────────
        transcript_text = transcription.transcript if transcription else sample.reference_transcript
        try:
            system_prompt = (
                "You are a helpful voice assistant. Answer the user's question concisely "
                "and accurately based only on the provided context.\n\n"
                f"Context:\n{sample.context}" if sample.context else
                "You are a helpful voice assistant. Answer concisely and accurately."
            )
            with timer("llm") as t:
                llm_resp = self.llm.generate(
                    prompt=transcript_text,
                    system=system_prompt,
                    max_tokens=256,
                )
            t_llm_ms = t["ms"]
            result.llm_response = llm_resp.response
            result.llm_latency_ms = llm_resp.latency_ms
        except Exception as e:
            logger.error(f"  LLM generation failed for {sample.sample_id}: {e}")
            result.errors.append(f"llm: {e}")
            result.llm_response = ""

        # ── Step 3: WER ──────────────────────────────────────────────────────
        if transcription is not None:
            try:
                result.wer = self.wer_calc.compute(
                    reference=sample.reference_transcript,
                    hypothesis=transcription.transcript,
                )
            except Exception as e:
                logger.warning(f"  WER failed: {e}")
                result.errors.append(f"wer: {e}")

        # ── Step 4: Semantic similarity ──────────────────────────────────────
        if result.llm_response:
            try:
                result.similarity = self.similarity_calc.compute(
                    text_a=sample.reference_response,
                    text_b=result.llm_response,
                )
            except Exception as e:
                logger.warning(f"  Similarity failed: {e}")
                result.errors.append(f"similarity: {e}")

        # ── Step 5: Hallucination detection ──────────────────────────────────
        if result.llm_response:
            context = sample.context or sample.reference_transcript
            try:
                result.hallucination = self.hallucination_detector.detect(
                    response=result.llm_response,
                    context=context,
                    reference=sample.reference_response,
                )
            except Exception as e:
                logger.warning(f"  Hallucination detection failed: {e}")
                result.errors.append(f"hallucination: {e}")

        # ── Step 6: Record latency ────────────────────────────────────────────
        total_ms = t_transcription_ms + t_llm_ms
        latency_record = LatencyRecord(
            label=sample.sample_id,
            total_ms=total_ms,
            transcription_ms=t_transcription_ms,
            llm_ms=t_llm_ms,
            audio_duration_s=transcription.duration_seconds if transcription else 0.0,
        )
        result.latency = latency_record
        self.latency_stats.add(latency_record)

        return result

    def _aggregate(self, results: list[SampleResult]) -> dict:
        """Compute aggregate metrics across all sample results."""
        wer_results   = [r.wer for r in results if r.wer is not None]
        sim_results   = [r.similarity for r in results if r.similarity is not None]
        hall_results  = [r.hallucination for r in results if r.hallucination is not None]
        error_counts  = sum(1 for r in results if r.errors)

        return {
            "wer": self.wer_calc.aggregate(wer_results) if wer_results else {},
            "semantic_similarity": self.similarity_calc.aggregate(sim_results) if sim_results else {},
            "hallucination": self.hallucination_detector.aggregate(hall_results) if hall_results else {},
            "latency": self.latency_stats.compute(),
            "errors": {
                "samples_with_errors": error_counts,
                "error_rate": round(error_counts / max(len(results), 1), 4),
            },
        }