"""
Eval harness — ties a model adapter + dataset + the right metric into one
scorecard, prints it, and saves a JSON report under eval/reports/.

Guiding rule: no model is trusted without a score, and no score is trusted
without first passing the label-orientation calibration probe.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics
from eval.datasets import Dataset
from eval.models import HFTextClassifier
from eval.tasks import TaskConfig

REPORTS_DIR = Path(__file__).parent / "reports"

# Headline numbers from the prior generic fake-news ONNX classifier (see
# eval/README.md) — printed alongside the RAG result for direct comparison.
_CLASSIFIER_BASELINE = {
    "pubhealth": "misinfo precision 0.557 / recall 0.925 / F1 0.695 (n=1200)",
    "sample": "misinfo precision 0.50 (chance — collapsed to all-misinfo)",
}


def run_classification(task: TaskConfig, dataset: Dataset,
                       save: bool = True) -> dict:
    """Score a classification task (misinfo/sentiment) and return the report."""
    print(f"\n{'=' * 72}")
    print(f"EVAL  task={task.name}  model={task.model_id}")
    print(f"      dataset={dataset.name}  n={len(dataset)}  "
          f"labels={dataset.labels}")
    print('=' * 72)

    t0 = time.perf_counter()
    model = HFTextClassifier(task.model_id, label_map=task.label_map)
    load_s = time.perf_counter() - t0

    # ── 1. Calibration probe (orientation guard) ──────────────────────────────
    cal = model.calibrate(task.sentinels, positive_label=task.positive_label,
                          threshold=task.threshold)
    print(f"\nCalibration (label-orientation check): "
          f"{cal['correct']}/{cal['n']} sentinels correct")
    for p in cal["probes"]:
        mark = "ok " if p["correct"] else "MISS"
        print(f"  [{mark}] expected={p['expected']:<9} got={p['predicted']:<9} "
              f"{p['scores']}  | {p['text']}")
    if not cal["orientation_ok"]:
        print("\n  [!] ORIENTATION SUSPECT -- the label map may be flipped, OR the "
              "model is degenerate (predicts one class regardless of input). "
              "Inspect the prediction spread below before trusting any metric.")

    # ── 2. Score the dataset ───────────────────────────────────────────────────
    t1 = time.perf_counter()
    preds = model.predict(dataset.texts(), positive_label=task.positive_label,
                          threshold=task.threshold)
    infer_s = time.perf_counter() - t1
    y_pred = [p.label for p in preds]
    y_true = dataset.gold()

    report = metrics.classification_report(
        y_true, y_pred, labels=dataset.labels,
        positive_label=task.positive_label,
    )

    print(f"\nResults  (load {load_s:.1f}s, inference {infer_s:.1f}s for "
          f"{len(dataset)} ex):")
    print(report.pretty())
    print(f"\n  HEADLINE: {report.headline()}")
    print(f"  GO-LIVE TARGET ({task.target_metric}): {task.target}")
    print(_verdict(task, report))

    out = {
        "task": task.name,
        "model_id": task.model_id,
        "dataset": dataset.name,
        "n": len(dataset),
        "threshold": task.threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "calibration": cal,
        "metrics": report.to_dict(),
        "target": {"metric": task.target_metric, "description": task.target},
        "timing_seconds": {"model_load": round(load_s, 2),
                           "inference": round(infer_s, 2)},
        "notes": task.notes,
    }
    if save:
        path = _save(out)
        print(f"\n  report saved -> {path}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# RAG misinfo evaluation — the retrieval-grounded replacement for the classifier
# ──────────────────────────────────────────────────────────────────────────────

# verdict → binary label. Precision-oriented (the product flags for human review,
# so we only call something misinfo when evidence actively *contradicts* it).
_VERDICT_TO_LABEL = {
    "contradicted": "misinfo",
    "supported": "reliable",
    "unsupported": "reliable",       # corpus gap → don't cry wolf on unseen claims
    "not_health_claim": "reliable",
    "unverified": "reliable",        # LLM down (shouldn't happen mid-eval)
}


def run_misinfo_rag(dataset: Dataset, max_examples: int | None = None,
                    k: int = 4, save: bool = True) -> dict:
    """
    Score the RAG misinfo checker (app/rag.check_claim) on a misinfo dataset and
    print it next to the old classifier's number. Each example is one LLM call, so
    keep max_examples small for benchmarks (the curated sample is the head-to-head).
    """
    from app.rag import check_claim

    examples = dataset.examples[:max_examples] if max_examples else dataset.examples
    print(f"\n{'=' * 72}")
    print(f"EVAL  task=misinfo  model=RAG (retrieve + LLM judge)")
    print(f"      dataset={dataset.name}  n={len(examples)}  k={k}")
    print('=' * 72)

    t0 = time.perf_counter()
    y_true, y_pred = [], []
    verdicts: Counter = Counter()
    for i, ex in enumerate(examples, 1):
        out = check_claim(ex.text, k=k)
        verdicts[out["verdict"]] += 1
        y_pred.append(_VERDICT_TO_LABEL.get(out["verdict"], "reliable"))
        y_true.append(ex.label)
        if i % 5 == 0 or i == len(examples):
            print(f"  scored {i}/{len(examples)}…", end="\r")
    infer_s = time.perf_counter() - t0

    report = metrics.classification_report(
        y_true, y_pred, labels=dataset.labels, positive_label="misinfo")

    print(f"\n\nResults  ({infer_s:.0f}s for {len(examples)} claims, "
          f"{infer_s / max(len(examples), 1):.1f}s/claim):")
    print(report.pretty())
    print(f"\n  verdict spread: {dict(verdicts)}")
    print(f"\n  HEADLINE (RAG):       {report.headline()}")
    name = dataset.name.lower()
    base_key = "sample" if ("sample" in name or "health_claims" in name) else "pubhealth"
    base = _CLASSIFIER_BASELINE[base_key]
    print(f"  prior classifier:     {base}")
    print("  mapping: contradicted→misinfo; supported/unsupported/not_claim→reliable "
          "(precision-oriented; recall is gated by corpus coverage).")

    out = {
        "task": "misinfo", "model_id": "rag (retrieve + LLM judge)",
        "dataset": dataset.name, "n": len(examples), "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": report.to_dict(),
        "verdict_spread": dict(verdicts),
        "classifier_baseline": base,
        "timing_seconds": {"inference": round(infer_s, 1)},
    }
    if save:
        path = _save(out)
        print(f"\n  report saved -> {path}")
    return out


def _verdict(task: TaskConfig, report: metrics.ClassificationReport) -> str:
    """Plain-language pass/fail against the task's headline metric, if numeric."""
    if report.positive_label is not None and report.positive is not None:
        val = getattr(report.positive, task.target_metric, None)
        metric_name = f"{report.positive_label} {task.target_metric}"
    else:
        # No single positive class (e.g. sentiment) → judge on macro-F1.
        val = report.macro_f1
        metric_name = "macro-F1"
    if val is None:
        return ""
    # crude bar parse: look for ">= X" in target text
    import re
    m = re.search(r">=\s*([0-9.]+)", task.target)
    if not m:
        return f"  VERDICT: {metric_name}={val:.3f} (no numeric bar parsed)"
    bar = float(m.group(1))
    status = "PASS" if val >= bar else "FAIL"
    return f"  VERDICT: {status} — {metric_name}={val:.3f} vs bar {bar:.2f}"


def _save(out: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"{out['task']}_{out['dataset']}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return path
