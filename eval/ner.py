"""
NER evaluation — entity-level F1 (Whisper/misinfo use sequence models; this is
token classification).

Scores d4data/biomedical-ner-all against NCBI-disease. The model uses the large
MACCROBAT schema (84 labels); NCBI-disease annotates a single `Disease` type, so
we collapse the model's disease tags onto `Disease` and everything else onto `O`,
then compute exact-match entity F1 (eval.metrics.entity_f1). Token accuracy is
deliberately avoided — 'O' dominates and would inflate the score.

Mapping note: only `Disease_disorder` → `Disease`. `Sign_symptom` is left as `O`
because NCBI-disease does not annotate symptoms; folding it in would manufacture
false positives the benchmark can't credit.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics
from eval.datasets import TokenDataset

warnings.filterwarnings("ignore")

REPORTS_DIR = Path(__file__).parent / "reports"

# model entity type -> canonical NCBI type (others drop to O)
_TYPE_MAP = {"Disease_disorder": "Disease"}


class HFTokenClassifier:
    def __init__(self, model_id: str, type_map: dict[str, str],
                 max_length: int = 512, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (AutoModelForTokenClassification
                      .from_pretrained(model_id).eval().to(self.device))
        self.id2label = dict(self.config.id2label)
        self.type_map = type_map
        self.max_length = max_length

    def _canon(self, raw_label: str) -> str:
        """Map a raw BIO label (e.g. 'B-Disease_disorder') to canonical BIO."""
        if raw_label == "O" or "-" not in raw_label:
            return "O"
        prefix, etype = raw_label.split("-", 1)
        canon = self.type_map.get(etype)
        return f"{prefix}-{canon}" if canon else "O"

    def predict_tags(self, tokens: list[str]) -> list[str]:
        """Word-aligned canonical BIO tags for one pre-tokenised sentence."""
        torch = self._torch
        enc = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                             truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        pred_ids = logits.argmax(-1).cpu().tolist()
        word_ids = enc.word_ids(0)

        tags = ["O"] * len(tokens)
        seen = set()
        for sub_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue                      # special token, or non-first subword
            seen.add(wid)
            raw = self.id2label.get(pred_ids[sub_idx], "O")
            tags[wid] = self._canon(raw)
        return tags


def evaluate(dataset: TokenDataset, model_id: str, target: str = "f1 >= 0.80",
             save: bool = True) -> dict:
    print(f"\n{'=' * 72}")
    print(f"EVAL  task=ner  model={model_id}")
    print(f"      dataset={dataset.name}  n={len(dataset)}  type_map={_TYPE_MAP}")
    print('=' * 72)

    t0 = time.perf_counter()
    clf = HFTokenClassifier(model_id, type_map=_TYPE_MAP)
    load_s = time.perf_counter() - t0

    # Sentinel: a known disease mention should be tagged.
    probe_tokens = "The patient was diagnosed with cystic fibrosis .".split()
    probe = clf.predict_tags(probe_tokens)
    caught = any(t.endswith("-Disease") for t in probe)
    print(f"\nCalibration: 'cystic fibrosis' detected = {caught}")
    print(f"  tokens: {probe_tokens}")
    print(f"  tags:   {probe}")
    if not caught:
        print("  [!] disease sentinel not detected — check the type map / model.")

    t1 = time.perf_counter()
    gold_seqs, pred_seqs = [], []
    for ex in dataset.examples:
        gold_seqs.append(ex.tags)
        pred_seqs.append(clf.predict_tags(ex.tokens))
    infer_s = time.perf_counter() - t1

    ef = metrics.entity_f1(gold_seqs, pred_seqs, relaxed=False)
    rel = metrics.entity_f1(gold_seqs, pred_seqs, relaxed=True)

    print(f"\nResults  (load {load_s:.1f}s, inference {infer_s:.1f}s for "
          f"{len(dataset)} sentences):")
    print(f"  exact span    precision={ef['precision']:.3f}  "
          f"recall={ef['recall']:.3f}  F1={ef['f1']:.3f}")
    print(f"  relaxed (overlap) precision={rel['precision']:.3f}  "
          f"recall={rel['recall']:.3f}  F1={rel['f1']:.3f}")
    print(f"  gold entities={ef['gold_entities']}  predicted={ef['pred_entities']}  "
          f"(exact tp={ef['tp']} fp={ef['fp']} fn={ef['fn']})")

    import re
    m = re.search(r">=\s*([0-9.]+)", target)
    bar = float(m.group(1)) if m else None
    if bar is not None:
        status = "PASS" if ef["f1"] >= bar else "FAIL"
        print(f"\n  GO-LIVE TARGET: entity F1 (exact) {target}")
        print(f"  VERDICT: {status} — exact F1={ef['f1']:.3f} "
              f"(relaxed {rel['f1']:.3f}) vs bar {bar:.2f}")

    out = {
        "task": "ner", "model_id": model_id, "dataset": dataset.name,
        "n": len(dataset), "type_map": _TYPE_MAP,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentinel_caught": caught,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in ef.items()},
        "metrics_relaxed": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in rel.items()},
        "target": target,
        "timing_seconds": {"model_load": round(load_s, 2),
                           "inference": round(infer_s, 2)},
    }
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = REPORTS_DIR / f"ner_{dataset.name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\n  report saved -> {path}")
    return out
