"""
Metrics — the right metric per task.

A single "accuracy" number is misleading for every task in this system, so each
task type computes and surfaces what actually matters:

  - misinfo   : precision/recall/F1 under class imbalance, with the *positive*
                (misinfo) class called out explicitly. Accuracy is reported but
                de-emphasised because a model that always predicts the majority
                class can score high accuracy while being useless.
  - triage    : recall/sensitivity on the emergency class (catching every
                emergency matters more than overall accuracy).
  - ner       : entity-level F1 (token-level is a stop-gap).
  - asr       : Word Error Rate (lower is better).

Everything here depends only on numpy + scikit-learn so it can run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


# ──────────────────────────────────────────────────────────────────────────────
# Classification (misinfo, sentiment, and the ML side of triage)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassMetric:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationReport:
    labels: list[str]
    per_class: list[ClassMetric]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray                      # rows = true, cols = predicted
    positive_label: str | None = None          # the class we actually care about
    n: int = 0
    # When a probability threshold was applied, how many examples were predicted
    # positive vs. how the model distributed its raw argmax. Helps spot a model
    # that has collapsed onto one class.
    pred_distribution: dict[str, int] = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    def _get(self, label: str) -> ClassMetric | None:
        return next((c for c in self.per_class if c.label == label), None)

    @property
    def positive(self) -> ClassMetric | None:
        return self._get(self.positive_label) if self.positive_label else None

    def headline(self) -> str:
        """One-line metric appropriate to the task."""
        if self.positive_label and self.positive is not None:
            p = self.positive
            return (f"{self.positive_label}: precision={p.precision:.3f} "
                    f"recall={p.recall:.3f} F1={p.f1:.3f} (n+={p.support})")
        return f"macro-F1={self.macro_f1:.3f} accuracy={self.accuracy:.3f}"

    def pretty(self) -> str:
        lines = []
        w = max((len(l) for l in self.labels), default=5) + 2
        lines.append(f"  {'label'.ljust(w)}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
        lines.append("  " + "-" * (w + 33))
        for c in self.per_class:
            star = " *" if c.label == self.positive_label else ""
            lines.append(f"  {c.label.ljust(w)}{c.precision:>8.3f}{c.recall:>8.3f}"
                         f"{c.f1:>8.3f}{c.support:>9}{star}")
        lines.append("  " + "-" * (w + 33))
        lines.append(f"  {'accuracy'.ljust(w)}{'':>8}{'':>8}{self.accuracy:>8.3f}{self.n:>9}")
        lines.append(f"  {'macro avg'.ljust(w)}{'':>16}{self.macro_f1:>8.3f}")
        lines.append(f"  {'weighted avg'.ljust(w)}{'':>16}{self.weighted_f1:>8.3f}")
        lines.append("")
        lines.append("  confusion (rows=true, cols=pred): "
                     + ", ".join(self.labels))
        for i, lab in enumerate(self.labels):
            row = "  ".join(str(int(x)) for x in self.confusion[i])
            lines.append(f"    {lab.ljust(w)} [ {row} ]")
        if self.pred_distribution:
            dist = ", ".join(f"{k}={v}" for k, v in self.pred_distribution.items())
            lines.append(f"  prediction spread: {dist}")
            collapsed = max(self.pred_distribution.values()) == self.n if self.n else False
            if collapsed:
                lines.append("  [!] model predicted a SINGLE class for every example "
                             "-- it is not discriminating.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "headline": self.headline(),
            "n": self.n,
            "labels": self.labels,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "positive_label": self.positive_label,
            "per_class": [
                {"label": c.label, "precision": round(c.precision, 4),
                 "recall": round(c.recall, 4), "f1": round(c.f1, 4),
                 "support": c.support}
                for c in self.per_class
            ],
            "confusion": self.confusion.astype(int).tolist(),
            "prediction_distribution": self.pred_distribution,
        }


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    positive_label: str | None = None,
) -> ClassificationReport:
    """
    Compute a per-class classification report.

    `labels` fixes the label ordering (so the confusion matrix is stable even
    when a model never predicts some class). `positive_label`, if given, is the
    class whose precision/recall we headline — for misinfo this is "misinfo".
    """
    labels = list(labels)
    y_true = list(y_true)
    y_pred = list(y_pred)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = [
        ClassMetric(labels[i], float(prec[i]), float(rec[i]), float(f1[i]), int(sup[i]))
        for i in range(len(labels))
    ]
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    total = int(sum(sup)) or 1
    weighted_f1 = float(np.sum(f1 * sup) / total)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    dist: dict[str, int] = {l: 0 for l in labels}
    for p in y_pred:
        dist[p] = dist.get(p, 0) + 1

    return ClassificationReport(
        labels=labels,
        per_class=per_class,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion=cm,
        positive_label=positive_label,
        n=len(y_true),
        pred_distribution=dist,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Safety-critical recall (triage emergency, distress detection)
# ──────────────────────────────────────────────────────────────────────────────

def safety_recall(y_true: Sequence[str], y_pred: Sequence[str], emergency_label: str) -> dict:
    """
    For triage/distress: the only number that matters first is "of the true
    emergencies, how many did we catch?". Misses (false negatives) are the
    dangerous error. Returns recall plus the raw miss count.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == emergency_label and p == emergency_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == emergency_label and p != emergency_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != emergency_label and p == emergency_label)
    pos = tp + fn
    return {
        "emergency_label": emergency_label,
        "recall": (tp / pos) if pos else float("nan"),
        "true_emergencies": pos,
        "caught": tp,
        "missed": fn,             # the dangerous number
        "over_escalations": fp,   # acceptable cost of high recall
    }


# ──────────────────────────────────────────────────────────────────────────────
# NER — entity-level F1 (exact span match)
# ──────────────────────────────────────────────────────────────────────────────

def bio_spans(tags: Sequence[str]) -> set[tuple[int, int, str]]:
    """
    Extract entity spans from a BIO tag sequence as (start, end_exclusive, type).
    'B-Disease' opens a span, 'I-Disease' extends it, 'O' (or a different type)
    closes it. Used for exact-match entity scoring.
    """
    spans: set[tuple[int, int, str]] = set()
    start = None
    etype = None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or tag.startswith("B-"):
            if start is not None:
                spans.add((start, i, etype))
                start, etype = None, None
            if tag.startswith("B-"):
                start, etype = i, tag[2:]
        elif tag.startswith("I-"):
            cur = tag[2:]
            if start is None:                  # I- with no B- → treat as new span
                start, etype = i, cur
            elif cur != etype:                 # type switch mid-span → close + open
                spans.add((start, i, etype))
                start, etype = i, cur
    return spans


def _overlap(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    """Same type and the [start,end) ranges intersect."""
    return a[2] == b[2] and a[0] < b[1] and b[0] < a[1]


def entity_f1(true_tag_seqs: Sequence[Sequence[str]],
              pred_tag_seqs: Sequence[Sequence[str]],
              relaxed: bool = False) -> dict:
    """
    Entity-level precision/recall/F1. Token accuracy is misleading ('O'
    dominates), so we score whole spans.

    relaxed=False : exact span match (the standard, strict metric).
    relaxed=True  : a predicted span counts if it *overlaps* a gold span of the
                    same type — tolerates boundary disagreements, which dominate
                    cross-dataset eval where annotation guidelines differ.
    """
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(true_tag_seqs, pred_tag_seqs):
        gold = bio_spans(true_tags)
        pred = bio_spans(pred_tags)
        if not relaxed:
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        else:
            matched_gold = set()
            for p in pred:
                hit = next((g for g in gold if g not in matched_gold and _overlap(p, g)), None)
                if hit is not None:
                    tp += 1
                    matched_gold.add(hit)
                else:
                    fp += 1
            fn += len(gold - matched_gold)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "gold_entities": tp + fn, "pred_entities": tp + fp}


# ──────────────────────────────────────────────────────────────────────────────
# ASR — Word Error Rate (Whisper)
# ──────────────────────────────────────────────────────────────────────────────

def wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate via Levenshtein distance over word tokens.
    (edits) / (reference words). Lower is better; 0.0 = perfect.
    """
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(h)]) / len(r)


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    pairs = list(zip(references, hypotheses))
    scores = [wer(r, h) for r, h in pairs]
    return {
        "wer_mean": float(np.mean(scores)) if scores else float("nan"),
        "wer_median": float(np.median(scores)) if scores else float("nan"),
        "n": len(pairs),
    }
