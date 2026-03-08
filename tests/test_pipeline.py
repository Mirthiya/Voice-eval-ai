"""
tests/test_pipeline.py — Comprehensive test suite for the Voice AI eval pipeline.

Coverage:
  - WER calculation (unit + edge cases)
  - Semantic similarity (unit + both methods)
  - Hallucination detection (unit + signal fusion)
  - Latency measurement (unit + aggregation)
  - Transcription (mock path)
  - LLM client (mock path)
  - Full pipeline integration (end-to-end with mocks)
  - Determinism guarantees
  - Report generation
"""

import os
import sys
import json
import time
import tempfile
import unittest
import wave
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluator.config import EvalConfig, set_global_seed
from evaluator.metrics.wer import WERCalculator, _normalize, _levenshtein
from evaluator.metrics.latency import LatencyRecord, LatencyStats, timer, timed
from evaluator.metrics.semantic_similarity import (
    SemanticSimilarityCalculator, _tfidf_cosine, _tokenize
)
from evaluator.metrics.hallucination import (
    HallucinationDetector, _extract_entities, _entity_overlap_score
)
from evaluator.transcription.whisper_client import WhisperTranscriber
from evaluator.llm.ollama_client import OllamaClient
from evaluator.pipeline import EvaluationPipeline, EvalSample, SampleResult
from evaluator.reporter import ReportWriter


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16000) -> str:
    """Create a minimal valid WAV file for testing."""
    n_frames = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return path


def make_sample(sample_id: str = "s001", audio_path: str = "/tmp/test.wav") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        audio_path=audio_path,
        reference_transcript="hello how are you today",
        reference_response="I am doing well thank you",
        context="The user greeted the assistant.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def test_defaults_are_sensible(self):
        cfg = EvalConfig()
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.ollama_temperature, 0.0)
        self.assertFalse(cfg.whisper_fp16)

    def test_fingerprint_is_deterministic(self):
        cfg1 = EvalConfig(seed=42)
        cfg2 = EvalConfig(seed=42)
        self.assertEqual(cfg1.fingerprint(), cfg2.fingerprint())

    def test_different_seeds_give_different_fingerprints(self):
        cfg1 = EvalConfig(seed=1)
        cfg2 = EvalConfig(seed=2)
        self.assertNotEqual(cfg1.fingerprint(), cfg2.fingerprint())

    def test_to_dict_is_serialisable(self):
        cfg = EvalConfig()
        d = cfg.to_dict()
        json.dumps(d)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalization(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(_normalize("Hello World"), "hello world")

    def test_punctuation_removed(self):
        result = _normalize("Hello, world!")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)

    def test_contraction_expansion(self):
        self.assertIn("do not", _normalize("don't"))
        self.assertIn("will not", _normalize("won't"))

    def test_whitespace_collapsed(self):
        result = _normalize("hello   world")
        self.assertEqual(result, "hello world")


class TestLevenshtein(unittest.TestCase):

    def test_identical(self):
        dist, s, d, i = _levenshtein(["a", "b"], ["a", "b"])
        self.assertEqual(dist, 0)
        self.assertEqual(s, 0)
        self.assertEqual(d, 0)
        self.assertEqual(i, 0)

    def test_one_substitution(self):
        dist, s, d, i = _levenshtein(["hello", "world"], ["hello", "earth"])
        self.assertEqual(s, 1)
        self.assertEqual(d, 0)
        self.assertEqual(i, 0)

    def test_one_deletion(self):
        dist, s, d, i = _levenshtein(["a", "b", "c"], ["a", "c"])
        self.assertEqual(d, 1)

    def test_one_insertion(self):
        dist, s, d, i = _levenshtein(["a", "c"], ["a", "b", "c"])
        self.assertEqual(i, 1)

    def test_empty_reference(self):
        dist, s, d, i = _levenshtein([], ["a", "b"])
        self.assertEqual(i, 2)

    def test_empty_hypothesis(self):
        dist, s, d, i = _levenshtein(["a", "b"], [])
        self.assertEqual(d, 2)


class TestWERCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = WERCalculator(normalize=True)

    def test_perfect_transcript(self):
        result = self.calc.compute("hello world", "hello world")
        self.assertAlmostEqual(result.wer, 0.0)

    def test_complete_mismatch(self):
        result = self.calc.compute("hello world", "goodbye universe")
        self.assertGreater(result.wer, 0)

    def test_one_word_error(self):
        result = self.calc.compute("hello world today", "hello earth today")
        self.assertAlmostEqual(result.substitutions, 1)

    def test_wer_formula(self):
        result = self.calc.compute("a b c d", "a b e d")
        self.assertAlmostEqual(result.wer, 1 / 4, places=4)

    def test_empty_reference_no_error(self):
        result = self.calc.compute("", "")
        self.assertAlmostEqual(result.wer, 0.0)

    def test_empty_hypothesis(self):
        result = self.calc.compute("hello world", "")
        self.assertGreater(result.wer, 0)

    def test_contractions_normalized(self):
        r1 = self.calc.compute("I will not go", "I won't go")
        self.assertAlmostEqual(r1.wer, 0.0, places=3)

    def test_cer_perfect(self):
        result = self.calc.compute("hello", "hello")
        self.assertAlmostEqual(result.cer, 0.0)

    def test_aggregate_corpus_wer(self):
        pairs = [
            ("hello world", "hello world"),
            ("foo bar baz", "foo bar qux"),
        ]
        results = self.calc.batch_compute(pairs)
        agg = self.calc.aggregate(results)
        self.assertIn("corpus_wer", agg)
        self.assertAlmostEqual(agg["corpus_wer"], 1 / 5, places=3)

    def test_determinism(self):
        r1 = self.calc.compute("hello world", "hello earth")
        r2 = self.calc.compute("hello world", "hello earth")
        self.assertEqual(r1.wer, r2.wer)
        self.assertEqual(r1.substitutions, r2.substitutions)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LATENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatency(unittest.TestCase):

    def test_timer_context_manager(self):
        with timer("test") as t:
            time.sleep(0.01)
        self.assertGreater(t["ms"], 5)

    def test_timed_decorator(self):
        def slow_fn(x):
            time.sleep(0.01)
            return x * 2
        result, ms = timed(slow_fn)(5)
        self.assertEqual(result, 10)
        self.assertGreater(ms, 5)

    def test_latency_record_rtf(self):
        record = LatencyRecord(
            label="test",
            total_ms=2000.0,
            audio_duration_s=4.0,
        )
        self.assertAlmostEqual(record.real_time_factor, 0.5, places=3)

    def test_rtf_zero_when_no_audio(self):
        record = LatencyRecord(label="test", total_ms=100.0, audio_duration_s=0.0)
        self.assertEqual(record.real_time_factor, 0.0)

    def test_latency_stats_aggregation(self):
        stats = LatencyStats(percentiles=[50, 90])
        for ms in [100, 200, 300, 400, 500]:
            stats.add(LatencyRecord(label="x", total_ms=float(ms)))
        result = stats.compute()
        self.assertAlmostEqual(result["total_ms"]["mean"], 300.0)
        self.assertAlmostEqual(result["total_ms"]["median"], 300.0)
        self.assertIn("p50", result["percentiles_ms"])
        self.assertIn("p90", result["percentiles_ms"])

    def test_percentile_p50(self):
        stats = LatencyStats(percentiles=[50])
        for ms in [10, 20, 30, 40, 50]:
            stats.add(LatencyRecord(label="x", total_ms=float(ms)))
        result = stats.compute()
        self.assertAlmostEqual(result["percentiles_ms"]["p50"], 30.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SEMANTIC SIMILARITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenize(unittest.TestCase):

    def test_basic(self):
        tokens = _tokenize("Hello, World!")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)

    def test_empty(self):
        self.assertEqual(_tokenize(""), [])


class TestTFIDFCosine(unittest.TestCase):

    def test_identical(self):
        score = _tfidf_cosine("hello world", "hello world")
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_unrelated(self):
        score = _tfidf_cosine("apple orange banana", "quantum physics neutron")
        self.assertLess(score, 0.3)

    def test_partial_overlap(self):
        score = _tfidf_cosine("hello world", "hello universe")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_strings(self):
        score = _tfidf_cosine("", "hello")
        self.assertEqual(score, 0.0)


class TestSemanticSimilarityCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = SemanticSimilarityCalculator(threshold=0.5)
        self.calc._model = None
        self.calc._method = "tfidf"

    def test_exact_match(self):
        result = self.calc.compute("hello world", "hello world")
        self.assertAlmostEqual(result.score, 1.0)
        self.assertEqual(result.method, "exact")

    def test_similar_sentences(self):
        result = self.calc.compute(
            "The cat sat on the mat",
            "The cat sat on the mat",
        )
        self.assertGreater(result.score, 0.8)

    def test_threshold_applied(self):
        result = self.calc.compute("apple", "neutron star")
        self.assertEqual(result.is_similar, result.score >= 0.5)

    def test_aggregate(self):
        pairs = [
            ("hello world", "hello world"),
            ("foo", "bar baz qux"),
        ]
        results = self.calc.batch_compute(pairs)
        agg = self.calc.aggregate(results)
        self.assertIn("mean_similarity", agg)
        self.assertIn("similar_rate", agg)

    def test_determinism(self):
        r1 = self.calc.compute("the dog is happy", "the dog is joyful")
        r2 = self.calc.compute("the dog is happy", "the dog is joyful")
        self.assertAlmostEqual(r1.score, r2.score, places=6)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HALLUCINATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntityExtraction(unittest.TestCase):

    def test_proper_nouns(self):
        entities = _extract_entities("Barack Obama visited Paris last June.")
        self.assertTrue(any("obama" in e or "barack" in e for e in entities))

    def test_numbers(self):
        entities = _extract_entities("The price is 42.5 dollars.")
        self.assertIn("42.5", entities)

    def test_percentages(self):
        entities = _extract_entities("Sales increased by 15%.")
        self.assertIn("15%", entities)

    def test_empty(self):
        entities = _extract_entities("")
        self.assertEqual(len(entities), 0)


class TestEntityOverlapScore(unittest.TestCase):

    def test_all_entities_in_context(self):
        score = _entity_overlap_score(
            response="Barack Obama was born in Hawaii.",
            context="Barack Obama was born in Hawaii in 1961.",
        )
        self.assertLess(score, 0.5)

    def test_entity_not_in_context(self):
        score = _entity_overlap_score(
            response="Napoleon conquered Moscow.",
            context="The user asked about modern history.",
        )
        self.assertGreater(score, 0.0)


class TestHallucinationDetector(unittest.TestCase):

    def setUp(self):
        self.detector = HallucinationDetector(llm_client=None, threshold=0.5)

    def test_grounded_response(self):
        result = self.detector.detect(
            response="The meeting is at 3pm.",
            context="The user said the meeting is at 3pm.",
        )
        self.assertIsInstance(result.is_hallucination, bool)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_signals_present(self):
        result = self.detector.detect(
            response="Napoleon was born in Corsica.",
            context="The user asked about World War II.",
        )
        self.assertIn("entity_overlap", result.signals)

    def test_aggregate(self):
        results = [
            self.detector.detect("hello", "hello world"),
            self.detector.detect("Napoleon invaded Egypt.", "user asked about weather"),
        ]
        agg = self.detector.aggregate(results)
        self.assertIn("hallucination_rate", agg)
        self.assertIn("num_samples", agg)
        self.assertEqual(agg["num_samples"], 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRANSCRIPTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWhisperTranscriber(unittest.TestCase):

    def setUp(self):
        self.transcriber = WhisperTranscriber()
        self.transcriber._available = False
        self.transcriber._model = None

    def test_mock_transcription(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            make_wav(wav_path)
            result = self.transcriber.transcribe(wav_path)
            self.assertIsInstance(result.transcript, str)
            self.assertGreater(len(result.transcript), 0)
            self.assertGreater(result.latency_ms, 0)
            self.assertEqual(result.model_used, "mock-transcriber")
        finally:
            os.unlink(wav_path)

    def test_mock_is_deterministic(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            make_wav(wav_path)
            r1 = self.transcriber.transcribe(wav_path)
            r2 = self.transcriber.transcribe(wav_path)
            self.assertEqual(r1.transcript, r2.transcript)
        finally:
            os.unlink(wav_path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.transcriber.transcribe("/nonexistent/path/audio.wav")

    def test_result_to_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            make_wav(wav_path)
            result = self.transcriber.transcribe(wav_path)
            d = result.to_dict()
            json.dumps(d)
        finally:
            os.unlink(wav_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LLM CLIENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaClient(unittest.TestCase):

    def setUp(self):
        self.client = OllamaClient()
        self.client._available = False

    def test_mock_response(self):
        resp = self.client.generate("What is the capital of France?")
        self.assertIsInstance(resp.response, str)
        self.assertTrue(resp.is_mock)
        self.assertGreater(resp.latency_ms, 0)

    def test_mock_is_deterministic(self):
        r1 = self.client.generate("Hello world")
        r2 = self.client.generate("Hello world")
        self.assertEqual(r1.response, r2.response)

    def test_hallucination_prompt_triggers_json(self):
        resp = self.client.generate("Check hallucination in this text")
        self.assertIn("hallucination", resp.response.lower())

    def test_response_to_dict(self):
        resp = self.client.generate("test prompt")
        d = resp.to_dict()
        json.dumps(d)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. REPORTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportWriter(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.writer = ReportWriter(report_dir=self.tmpdir)

    def test_save_and_load(self):
        report = {"run_id": "test", "metrics": {"wer": 0.1}}
        path = self.writer.save(report)
        loaded = self.writer.load(path)
        self.assertEqual(loaded["run_id"], "test")
        self.assertAlmostEqual(loaded["metrics"]["wer"], 0.1)

    def test_custom_path(self):
        path = os.path.join(self.tmpdir, "custom_report.json")
        report = {"run_id": "custom"}
        saved_path = self.writer.save(report, path=path)
        self.assertTrue(os.path.exists(saved_path))

    def test_summary_string(self):
        report = {
            "run_id": "test123",
            "num_samples": 5,
            "wall_time_seconds": 12.3,
            "config_fingerprint": "abc123",
            "aggregate_metrics": {
                "wer": {"corpus_wer": 0.05, "mean_sentence_wer": 0.06, "mean_cer": 0.02},
                "semantic_similarity": {"mean_similarity": 0.87, "similar_rate": 0.9, "method": "tfidf"},
                "hallucination": {"hallucination_rate": 0.1, "mean_confidence": 0.2,
                                  "num_hallucinations": 1, "num_samples": 5},
                "latency": {"total_ms": {"mean": 250, "median": 230, "stdev": 30},
                            "percentiles_ms": {"p50": 230, "p90": 310},
                            "real_time_factor": {"mean": 0.3}},
                "errors": {"error_rate": 0.0, "samples_with_errors": 0},
            },
        }
        summary = self.writer.summary(report)
        self.assertIn("WER", summary)
        self.assertIn("Hallucination", summary)
        self.assertIn("Latency", summary)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg = EvalConfig(report_dir=self.tmpdir, seed=42)
        self.pipeline = EvaluationPipeline(self.cfg)
        self.pipeline.transcriber._available = False
        self.pipeline.transcriber._model = None
        self.pipeline.llm._available = False
        self.pipeline.similarity_calc._model = None
        self.pipeline.similarity_calc._method = "tfidf"

        self.wav1 = os.path.join(self.tmpdir, "s001.wav")
        self.wav2 = os.path.join(self.tmpdir, "s002.wav")
        make_wav(self.wav1)
        make_wav(self.wav2, duration_s=2.0)

    def _make_samples(self):
        return [
            EvalSample(
                sample_id="s001",
                audio_path=self.wav1,
                reference_transcript="hello how are you today",
                reference_response="I am doing well thank you",
                context="Greeting context",
            ),
            EvalSample(
                sample_id="s002",
                audio_path=self.wav2,
                reference_transcript="what is the weather like",
                reference_response="The weather is sunny today",
                context="Weather information: sunny skies expected.",
            ),
        ]

    def test_pipeline_runs_without_error(self):
        samples = self._make_samples()
        report = self.pipeline.run(samples, run_id="test_run")
        self.assertIn("run_id", report)
        self.assertEqual(report["run_id"], "test_run")
        self.assertEqual(report["num_samples"], 2)

    def test_report_has_all_metric_sections(self):
        samples = self._make_samples()
        report = self.pipeline.run(samples)
        agg = report["aggregate_metrics"]
        self.assertIn("wer", agg)
        self.assertIn("semantic_similarity", agg)
        self.assertIn("hallucination", agg)
        self.assertIn("latency", agg)
        self.assertIn("errors", agg)

    def test_sample_results_count(self):
        samples = self._make_samples()
        report = self.pipeline.run(samples)
        self.assertEqual(len(report["sample_results"]), 2)

    def test_report_is_json_serialisable(self):
        samples = self._make_samples()
        report = self.pipeline.run(samples)
        json.dumps(report)

    def test_sample_validation_fails_on_missing_id(self):
        bad_sample = EvalSample(
            sample_id="",
            audio_path=self.wav1,
            reference_transcript="test",
            reference_response="test",
        )
        with self.assertRaises(ValueError):
            self.pipeline.run([bad_sample])

    def test_save_report(self):
        samples = self._make_samples()
        report = self.pipeline.run(samples, run_id="save_test")
        path = self.pipeline.save_report(report)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["run_id"], "save_test")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminism(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wav = os.path.join(self.tmpdir, "det.wav")
        make_wav(self.wav)

    def _make_pipeline(self) -> EvaluationPipeline:
        cfg = EvalConfig(seed=42, report_dir=self.tmpdir)
        p = EvaluationPipeline(cfg)
        p.transcriber._available = False
        p.transcriber._model = None
        p.llm._available = False
        p.similarity_calc._model = None
        return p

    def _make_samples(self, wav_path):
        return [EvalSample(
            sample_id="d001",
            audio_path=wav_path,
            reference_transcript="hello world",
            reference_response="hi there",
            context="test context",
        )]

    def test_wer_is_deterministic(self):
        calc = WERCalculator()
        for _ in range(5):
            r = calc.compute("hello world today", "hello earth today")
            self.assertAlmostEqual(r.wer, 1 / 3, places=4)

    def test_pipeline_deterministic_across_runs(self):
        p1 = self._make_pipeline()
        p2 = self._make_pipeline()
        samples = self._make_samples(self.wav)
        r1 = p1.run(samples, run_id="run1")
        r2 = p2.run(samples, run_id="run1")
        self.assertEqual(
            r1["aggregate_metrics"]["wer"].get("corpus_wer"),
            r2["aggregate_metrics"]["wer"].get("corpus_wer"),
        )
        t1 = r1["sample_results"][0]["transcription"]
        t2 = r2["sample_results"][0]["transcription"]
        if t1 and t2:
            self.assertEqual(t1["transcript"], t2["transcript"])

    def test_tfidf_similarity_deterministic(self):
        calc = SemanticSimilarityCalculator()
        calc._model = None
        scores = [calc.compute("the cat sat", "the dog stood").score for _ in range(5)]
        self.assertEqual(len(set(round(s, 8) for s in scores)), 1)

    def test_config_fingerprint_stability(self):
        cfg = EvalConfig()
        fps = [cfg.fingerprint() for _ in range(10)]
        self.assertEqual(len(set(fps)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)