"""
metrics/wer.py — Word Error Rate (WER) calculator.

WER = (S + D + I) / N
  S = substitutions
  D = deletions
  I = insertions
  N = number of words in reference

Uses dynamic programming (Wagner-Fischer algorithm) for exact edit distance.
Pure Python — no external dependencies for this metric.
"""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass


@dataclass
class WERResult:
    """Detailed WER breakdown."""
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    hypothesis_words: int
    reference_text: str
    hypothesis_text: str

    def to_dict(self) -> dict:
        return {
            "wer": round(self.wer, 4),
            "cer": round(self.cer, 4),
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_words": self.reference_words,
            "hypothesis_words": self.hypothesis_words,
        }


def _normalize(text: str) -> str:
    """
    Normalize text for fair WER comparison.

    Steps:
      1. Unicode NFC normalization
      2. Lowercase
      3. Expand contractions (basic)
      4. Remove punctuation
      5. Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    contractions = {
        "won't": "will not", "can't": "cannot", "don't": "do not",
        "isn't": "is not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "it's": "it is", "that's": "that is", "there's": "there is",
        "they're": "they are", "they've": "they have", "they'll": "they will",
        "we're": "we are", "we've": "we have", "we'll": "we will",
        "you're": "you are", "you've": "you have", "you'll": "you will",
        "let's": "let us", "couldn't": "could not", "wouldn't": "would not",
        "shouldn't": "should not",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """
    Wagner-Fischer dynamic programming edit distance.

    Returns (distance, substitutions, deletions, insertions).
    """
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_d, sub_s, sub_del, sub_ins = dp[i - 1][j - 1]
                del_d, del_s, del_del, del_ins = dp[i - 1][j]
                ins_d, ins_s, ins_del, ins_ins = dp[i][j - 1]

                best = min(
                    (sub_d + 1, sub_s + 1, sub_del,     sub_ins),
                    (del_d + 1, del_s,     del_del + 1, del_ins),
                    (ins_d + 1, ins_s,     ins_del,     ins_ins + 1),
                )
                dp[i][j] = best

    dist, subs, dels, ins = dp[n][m]
    return dist, subs, dels, ins


class WERCalculator:
    """
    Compute Word Error Rate and Character Error Rate.

    Both metrics are computed with the same normalization pipeline
    to ensure consistency and reproducibility.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute(self, reference: str, hypothesis: str) -> WERResult:
        """
        Compute WER between reference and hypothesis strings.

        Parameters
        ----------
        reference : str
            Ground-truth transcript.
        hypothesis : str
            ASR-produced transcript to evaluate.

        Returns
        -------
        WERResult
        """
        ref_norm = _normalize(reference) if self.normalize else reference.lower()
        hyp_norm = _normalize(hypothesis) if self.normalize else hypothesis.lower()

        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        if not ref_words:
            return WERResult(
                wer=0.0 if not hyp_words else 1.0,
                cer=0.0,
                substitutions=0, deletions=0, insertions=len(hyp_words),
                reference_words=0, hypothesis_words=len(hyp_words),
                reference_text=reference, hypothesis_text=hypothesis,
            )

        _, subs, dels, ins = _levenshtein(ref_words, hyp_words)
        wer = (subs + dels + ins) / len(ref_words)

        ref_chars = list(ref_norm.replace(" ", ""))
        hyp_chars = list(hyp_norm.replace(" ", ""))
        cer_dist, _, _, _ = _levenshtein(ref_chars, hyp_chars)
        cer = cer_dist / max(len(ref_chars), 1)

        return WERResult(
            wer=wer,
            cer=cer,
            substitutions=subs,
            deletions=dels,
            insertions=ins,
            reference_words=len(ref_words),
            hypothesis_words=len(hyp_words),
            reference_text=reference,
            hypothesis_text=hypothesis,
        )

    def batch_compute(self, pairs: list[tuple[str, str]]) -> list[WERResult]:
        """Compute WER for a list of (reference, hypothesis) pairs."""
        return [self.compute(ref, hyp) for ref, hyp in pairs]

    def aggregate(self, results: list[WERResult]) -> dict:
        """Compute corpus-level WER (pool all words, not average of sentence WERs)."""
        total_s = sum(r.substitutions for r in results)
        total_d = sum(r.deletions for r in results)
        total_i = sum(r.insertions for r in results)
        total_n = sum(r.reference_words for r in results)
        total_cer = sum(r.cer for r in results) / max(len(results), 1)

        corpus_wer = (total_s + total_d + total_i) / max(total_n, 1)
        wers = [r.wer for r in results]

        return {
            "corpus_wer": round(corpus_wer, 4),
            "mean_sentence_wer": round(sum(wers) / len(wers), 4) if wers else 0.0,
            "min_wer": round(min(wers), 4) if wers else 0.0,
            "max_wer": round(max(wers), 4) if wers else 0.0,
            "mean_cer": round(total_cer, 4),
            "total_substitutions": total_s,
            "total_deletions": total_d,
            "total_insertions": total_i,
            "total_reference_words": total_n,
            "num_samples": len(results),
        }