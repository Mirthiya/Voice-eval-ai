"""
transcription/whisper_client.py — Deterministic Whisper transcription pipeline.

Wraps OpenAI Whisper with:
  - Fixed decoding parameters (no randomness)
  - Latency measurement
  - Confidence / word-level timestamps
  - Fallback to mock transcription when Whisper is unavailable
"""

from __future__ import annotations
import time
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Output of the transcription pipeline for a single audio file."""
    audio_path: str
    transcript: str
    language: str
    duration_seconds: float
    latency_ms: float
    confidence: float
    word_timestamps: list[dict] = field(default_factory=list)
    model_used: str = "whisper-base"
    audio_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "transcript": self.transcript,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 3),
            "latency_ms": round(self.latency_ms, 2),
            "confidence": round(self.confidence, 4),
            "word_timestamps": self.word_timestamps,
            "model_used": self.model_used,
            "audio_fingerprint": self.audio_fingerprint,
        }


class WhisperTranscriber:
    """
    Deterministic Whisper transcription client.

    Determinism guarantees:
      - fp16=False  → identical float32 arithmetic on CPU
      - beam_size fixed → no sampling
      - temperature=0  → greedy decoding
      - same audio bytes → same transcript
    """

    def __init__(self, model_name: str = "base", language: Optional[str] = None,
                 fp16: bool = False, beam_size: int = 5):
        self.model_name = model_name
        self.language = language
        self.fp16 = fp16
        self.beam_size = beam_size
        self._model = None
        self._available = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load Whisper; gracefully fall back if unavailable."""
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
            self._available = True
            logger.info("Whisper loaded successfully")
        except ImportError:
            logger.warning("openai-whisper not installed — using mock transcriber")
        except Exception as e:
            logger.warning(f"Whisper load failed ({e}) — using mock transcriber")

    @staticmethod
    def _audio_fingerprint(audio_path: str) -> str:
        """SHA-256 fingerprint of raw audio bytes."""
        try:
            with open(audio_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"

    @staticmethod
    def _audio_duration(audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import wave
            with wave.open(audio_path, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Parameters
        ----------
        audio_path : str
            Path to WAV/MP3/FLAC audio file.

        Returns
        -------
        TranscriptionResult
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        fingerprint = self._audio_fingerprint(audio_path)
        duration = self._audio_duration(audio_path)

        if self._available and self._model is not None:
            return self._transcribe_whisper(audio_path, fingerprint, duration)
        else:
            return self._transcribe_mock(audio_path, fingerprint, duration)

    def _transcribe_whisper(self, audio_path: str, fingerprint: str,
                            duration: float) -> TranscriptionResult:
        """Run real Whisper transcription with deterministic settings."""
        import whisper

        t0 = time.perf_counter()
        result = self._model.transcribe(
            audio_path,
            language=self.language,
            fp16=self.fp16,
            beam_size=self.beam_size,
            temperature=0.0,
            word_timestamps=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        word_timestamps = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                word_timestamps.append({
                    "word": word.get("word", "").strip(),
                    "start": round(word.get("start", 0), 3),
                    "end": round(word.get("end", 0), 3),
                    "probability": round(word.get("probability", 0), 4),
                })

        segments = result.get("segments", [])
        if segments:
            avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
        else:
            confidence = 0.0

        return TranscriptionResult(
            audio_path=audio_path,
            transcript=result.get("text", "").strip(),
            language=result.get("language", "en"),
            duration_seconds=duration,
            latency_ms=latency_ms,
            confidence=confidence,
            word_timestamps=word_timestamps,
            model_used=f"whisper-{self.model_name}",
            audio_fingerprint=fingerprint,
        )

    def _transcribe_mock(self, audio_path: str, fingerprint: str,
                         duration: float) -> TranscriptionResult:
        """
        Mock transcription for CI/testing without Whisper installed.
        Returns deterministic output based on filename hash.
        """
        seed_val = int(hashlib.md5(audio_path.encode()).hexdigest()[:8], 16)
        mock_phrases = [
            "Hello, how can I help you today?",
            "The weather forecast shows sunny skies.",
            "Please hold while I connect your call.",
            "Your appointment is confirmed for Thursday.",
            "I didn't quite catch that, could you repeat?",
        ]
        transcript = mock_phrases[seed_val % len(mock_phrases)]

        t0 = time.perf_counter()
        time.sleep(0.01)
        latency_ms = (time.perf_counter() - t0) * 1000

        return TranscriptionResult(
            audio_path=audio_path,
            transcript=transcript,
            language="en",
            duration_seconds=duration or 2.5,
            latency_ms=latency_ms,
            confidence=0.85,
            word_timestamps=[],
            model_used="mock-transcriber",
            audio_fingerprint=fingerprint,
        )