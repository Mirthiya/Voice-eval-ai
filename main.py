
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Voice AI Evaluation Pipeline CLI

Usage:
    python main.py --samples samples/demo.json --run-id demo_run
    python main.py --help
"""
import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from evaluator import EvaluationPipeline, EvalConfig
from evaluator.pipeline import EvalSample


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def load_samples(path: str) -> list[EvalSample]:
    """Load evaluation samples from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        samples.append(EvalSample(
            sample_id=item["sample_id"],
            audio_path=item["audio_path"],
            reference_transcript=item["reference_transcript"],
            reference_response=item["reference_response"],
            context=item.get("context", ""),
            metadata=item.get("metadata", {}),
        ))
    return samples


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        seed=args.seed,
        whisper_model=args.whisper_model,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        report_dir=args.report_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice AI Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples",       default="samples/demo.json",
                        help="Path to JSON samples file")
    parser.add_argument("--run-id",        default=None,
                        help="Unique run identifier (auto-generated if omitted)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Global random seed for determinism")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--ollama-host",   default="http://localhost:11434")
    parser.add_argument("--ollama-model",  default="llama3")
    parser.add_argument("--report-dir",    default="reports")
    parser.add_argument("--log-level",     default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--print-summary", action="store_true",
                        help="Print human-readable summary to stdout")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    logger.info(f"Loading samples from: {args.samples}")
    try:
        samples = load_samples(args.samples)
    except FileNotFoundError:
        logger.error(f"Samples file not found: {args.samples}")
        logger.info("Generating demo samples for demonstration...")
        samples = _make_demo_samples()

    logger.info(f"Loaded {len(samples)} evaluation samples")

    config = build_config(args)
    pipeline = EvaluationPipeline(config)

    report = pipeline.run(samples, run_id=args.run_id)

    report_path = pipeline.save_report(report)
    logger.info(f"Report saved → {report_path}")

    if args.print_summary:
        print(pipeline.reporter.summary(report))


def _make_demo_samples() -> list[EvalSample]:
    """Create demo samples for running without real audio files."""
    import tempfile
    import wave
    import struct

    tmpdir = tempfile.mkdtemp()

    def make_wav(path, duration_s=2.0, sample_rate=16000):
        n = int(duration_s * sample_rate)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
        return path

    return [
        EvalSample(
            sample_id="demo_001",
            audio_path=make_wav(os.path.join(tmpdir, "demo_001.wav")),
            reference_transcript="hello how are you today",
            reference_response="I am doing well, thank you for asking!",
            context="Casual greeting interaction.",
        ),
        EvalSample(
            sample_id="demo_002",
            audio_path=make_wav(os.path.join(tmpdir, "demo_002.wav"), 3.0),
            reference_transcript="what is the weather forecast for tomorrow",
            reference_response="Tomorrow will be sunny with a high of 72 degrees.",
            context="Weather forecast: sunny skies, high 72F.",
        ),
        EvalSample(
            sample_id="demo_003",
            audio_path=make_wav(os.path.join(tmpdir, "demo_003.wav"), 2.5),
            reference_transcript="please set a reminder for three pm",
            reference_response="I have set a reminder for 3:00 PM.",
            context="User requested a reminder.",
        ),
    ]


if __name__ == "__main__":
    main()