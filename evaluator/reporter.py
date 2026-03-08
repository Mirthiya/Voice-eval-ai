"""
reporter.py — JSON evaluation report writer.
"""

from __future__ import annotations
import json
import os
import datetime
import numpy as np
from pathlib import Path
from typing import Optional


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


class ReportWriter:

    def __init__(self, report_dir: str = "reports", indent: int = 2):
        self.report_dir = Path(report_dir)
        self.indent = indent
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: dict, path: Optional[str] = None) -> str:
        if path is None:
            run_id   = report.get("run_id", "unknown")
            ts       = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{run_id}_{ts}.json"
            path     = str(self.report_dir / filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=self.indent, ensure_ascii=False, cls=NumpyEncoder)

        return os.path.abspath(path)

    def load(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def summary(self, report: dict) -> str:
        agg = report.get("aggregate_metrics", {})
        wer = agg.get("wer", {})
        sim = agg.get("semantic_similarity", {})
        hal = agg.get("hallucination", {})
        lat = agg.get("latency", {})
        err = agg.get("errors", {})

        lines = [
            "=" * 60,
            f"  Voice AI Evaluation Report  —  Run: {report.get('run_id', '?')}",
            "=" * 60,
            f"  Samples   : {report.get('num_samples', 0)}",
            f"  Wall time : {report.get('wall_time_seconds', 0):.2f}s",
            f"  Config    : {report.get('config_fingerprint', '?')}",
            "",
            "  ── Word Error Rate (WER) ──────────────────────────",
            f"  Corpus WER     : {wer.get('corpus_wer', 'N/A')}",
            f"  Mean Sent. WER : {wer.get('mean_sentence_wer', 'N/A')}",
            f"  Mean CER       : {wer.get('mean_cer', 'N/A')}",
            "",
            "  ── Semantic Similarity ────────────────────────────",
            f"  Mean Score     : {sim.get('mean_similarity', 'N/A')}",
            f"  Similar Rate   : {sim.get('similar_rate', 'N/A')}",
            f"  Method         : {sim.get('method', 'N/A')}",
            "",
            "  ── Hallucination Rate ─────────────────────────────",
            f"  Rate           : {hal.get('hallucination_rate', 'N/A')}",
            f"  Mean Confidence: {hal.get('mean_confidence', 'N/A')}",
            f"  Flagged        : {hal.get('num_hallucinations', 'N/A')}/{hal.get('num_samples', 'N/A')}",
            "",
            "  ── Latency ────────────────────────────────────────",
        ]

        lat_total = lat.get("total_ms", {})
        if lat_total:
            lines += [
                f"  Mean Total     : {lat_total.get('mean', 'N/A')} ms",
                f"  Median Total   : {lat_total.get('median', 'N/A')} ms",
                f"  Stdev          : {lat_total.get('stdev', 'N/A')} ms",
            ]
        pct = lat.get("percentiles_ms", {})
        if pct:
            lines.append(
                "  Percentiles    : " +
                " | ".join(f"{k}={v}ms" for k, v in sorted(pct.items()))
            )
        rtf = lat.get("real_time_factor", {})
        if rtf:
            lines.append(f"  Mean RTF       : {rtf.get('mean', 'N/A')} (< 1.0 = faster than real-time)")

        lines += [
            "",
            "  ── Errors ─────────────────────────────────────────",
            f"  Error Rate     : {err.get('error_rate', 'N/A')}",
            f"  Samples w/ Err : {err.get('samples_with_errors', 0)}",
            "=" * 60,
        ]

        return "\n".join(lines)