# Voice AI Evaluation Pipeline

A production-grade, deterministic evaluation framework for Voice AI systems built with Python, Whisper, and Ollama.

## Features
- Audio transcription using OpenAI Whisper
- Word Error Rate (WER) and Character Error Rate (CER)
- Semantic similarity using sentence-transformers
- Hallucination detection (multi-signal fusion)
- Latency profiling with p50/p90/p95/p99 percentiles + RTF
- Deterministic scoring across all runs (seed=42)
- JSON evaluation reports with config fingerprint

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull LLM model
```bash
ollama pull llama3.2:1b
ollama serve
```

### 3. Run evaluation
```bash
py -3 main.py --print-summary
```

### 4. Run tests
```bash
py -3 -m unittest tests/test_pipeline.py -v
```

## Sample Output
```
============================================================
  Voice AI Evaluation Report  —  Run: a1767991
============================================================
  Samples   : 3
  Wall time : 29.87s
  ── Word Error Rate ──────────────────────────────────
  Corpus WER     : 0.3158
  Mean CER       : 0.1262
  ── Semantic Similarity ──────────────────────────────
  Mean Score     : 0.0836
  Method         : sentence_transformer
  ── Hallucination Rate ───────────────────────────────
  Rate           : 0.0
  Flagged        : 0/3
  ── Latency ──────────────────────────────────────────
  Mean Total     : 7839ms
  Mean RTF       : 1.47
  ── Errors ───────────────────────────────────────────
  Error Rate     : 0.0
============================================================
```

## Project Structure
```
Voice_eval_ai/
├── main.py                          # CLI entry point
├── requirements.txt
├── evaluator/
│   ├── config.py                    # EvalConfig + determinism settings
│   ├── pipeline.py                  # Main evaluation orchestrator
│   ├── reporter.py                  # JSON report writer
│   ├── llm/
│   │   └── ollama_client.py         # Ollama LLM client
│   ├── metrics/
│   │   ├── wer.py                   # WER + CER (Wagner-Fischer DP)
│   │   ├── latency.py               # Latency + RTF profiling
│   │   ├── semantic_similarity.py   # Cosine similarity
│   │   └── hallucination.py         # Multi-signal hallucination detector
│   └── transcription/
│       └── whisper_client.py        # Whisper transcription client
├── tests/
│   └── test_pipeline.py             # 71 unit + integration tests
└── samples/
    └── demo.json                    # Sample evaluation data
```

## Determinism Guarantees
| Component | Setting |
|-----------|---------|
| Global seed | `seed=42` (Python, NumPy) |
| Whisper | `fp16=False`, `temperature=0`, `beam_size=5` |
| Ollama | `temperature=0.0`, `seed=42`, `top_k=1` |
| Sample order | Sorted by `sample_id` before evaluation |
| Audit trail | SHA-256 config fingerprint in every report |

## Metrics
| Metric | Method | Notes |
|--------|--------|-------|
| WER | Wagner-Fischer DP | Corpus + sentence level |
| CER | Character-level edit distance | Included in WER output |
| Semantic Similarity | sentence-transformers cosine | TF-IDF fallback |
| Hallucination | LLM judge + entity overlap + length ratio | 3-signal fusion |
| Latency | Wall-clock timing | p50/p90/p95/p99 + RTF |

## Test Coverage
71 tests across 10 test classes:
- Unit tests: WER, latency, similarity, hallucination, transcription, LLM
- Integration tests: full pipeline end-to-end
- Determinism tests: 5 repeated runs produce identical scores

## Graceful Degradation
- No Whisper → deterministic mock transcription
- No Ollama → deterministic mock LLM responses
- No sentence-transformers → TF-IDF cosine similarity fallback
