# Voice AI Evaluation Pipeline

A Python-based evaluation framework for Voice AI systems.

## Features
- Audio transcription using OpenAI Whisper
- Word Error Rate (WER) and Character Error Rate (CER)
- Semantic similarity using sentence-transformers
- Hallucination detection (multi-signal fusion)
- Latency profiling with p50/p90/p95/p99 percentiles
- Deterministic scoring (seed=42 everywhere)
- JSON evaluation reports

## Setup
pip install -r requirements.txt
ollama pull llama3.2:1b

## Run
py -3 main.py --print-summary

## Run Tests
py -3 -m unittest tests/test_pipeline.py -v

## Project Structure
evaluator/
  config.py          - EvalConfig with determinism settings
  pipeline.py        - Main evaluation pipeline
  reporter.py        - JSON report writer
  llm/               - Ollama LLM client
  metrics/           - WER, latency, similarity, hallucination
  transcription/     - Whisper transcription client
tests/
  test_pipeline.py   - 71 unit + integration tests
samples/
  demo.json          - Sample evaluation data
reports/             - Auto-generated JSON reports
