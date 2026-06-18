"""
Model adapters for evaluation.

The production runtime serves these models via ONNX Runtime (see
app/core/model_registry.py). For evaluation we load the *same* HuggingFace
checkpoint with PyTorch and read its logits directly. ONNX is exported from this
exact checkpoint, so the scores are faithful — and this lets us score a model
BEFORE it has been exported to ONNX (the export pipeline isn't wired yet).

Label orientation is the dangerous failure mode: if "misinfo" and "reliable" are
swapped, a high recall number is actually a model that flags everything truthful.
Every adapter therefore takes an explicit `label_map` and exposes a
`calibrate()` probe that runs known-true / known-false sentinels so a flipped
mapping is caught loudly, not silently.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class Prediction:
    label: str                       # canonical task label after mapping
    score: float                     # probability of `label`
    raw_scores: dict[str, float]     # canonical label -> probability


class HFTextClassifier:
    """
    Wraps a HF sequence-classification checkpoint and maps its raw output labels
    (LABEL_0/LABEL_1/...) into the task's canonical label space.

    label_map: { raw_model_label : canonical_label }. Required and explicit —
    we never guess orientation.
    """

    def __init__(self, model_id: str, label_map: dict[str, str],
                 max_length: int = 512, device: str | None = None):
        import torch
        from transformers import (AutoTokenizer, AutoConfig,
                                   AutoModelForSequenceClassification)

        self.model_id = model_id
        self.max_length = max_length
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(model_id).eval().to(self.device))

        self.raw_id2label = dict(self.config.id2label)  # {0: 'LABEL_0', ...}

        missing = set(self.raw_id2label.values()) - set(label_map)
        if missing:
            raise ValueError(
                f"label_map for {model_id} is missing raw labels {sorted(missing)}. "
                f"Model emits {sorted(self.raw_id2label.values())}; map every one to "
                f"a canonical label.")
        self.label_map = label_map
        # canonical label space, stable order
        self.canonical_labels = sorted(set(label_map.values()))

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        """Return, per text, canonical_label -> summed probability."""
        torch = self._torch
        out: list[dict[str, float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                 padding=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                agg: dict[str, float] = {l: 0.0 for l in self.canonical_labels}
                for raw_id, p in enumerate(row):
                    canon = self.label_map[self.raw_id2label[raw_id]]
                    agg[canon] += float(p)
                out.append(agg)
        return out

    def predict(self, texts: list[str], positive_label: str | None = None,
                threshold: float | None = None, batch_size: int = 16) -> list[Prediction]:
        """
        Predict canonical labels. If `positive_label` + `threshold` are given,
        an example is positive when P(positive_label) >= threshold (lets us honour
        the production MISINFO_THRESHOLD). Otherwise argmax.
        """
        preds = []
        for probs in self.predict_proba(texts, batch_size=batch_size):
            if positive_label is not None and threshold is not None:
                if probs.get(positive_label, 0.0) >= threshold:
                    label = positive_label
                else:
                    others = {k: v for k, v in probs.items() if k != positive_label}
                    label = max(others, key=others.get) if others else positive_label
            else:
                label = max(probs, key=probs.get)
            preds.append(Prediction(label=label, score=probs[label], raw_scores=probs))
        return preds

    def calibrate(self, sentinels: list[tuple[str, str]],
                  positive_label: str | None = None,
                  threshold: float | None = None) -> dict:
        """
        Run known-answer sentinels to verify label orientation before trusting
        any benchmark number. `sentinels` = [(text, expected_canonical_label)].
        Returns the probes + whether the mapping looks correct.
        """
        texts = [t for t, _ in sentinels]
        preds = self.predict(texts, positive_label=positive_label, threshold=threshold)
        probes, correct = [], 0
        for (text, expected), pred in zip(sentinels, preds):
            ok = pred.label == expected
            correct += ok
            probes.append({
                "text": text, "expected": expected, "predicted": pred.label,
                "scores": {k: round(v, 4) for k, v in pred.raw_scores.items()},
                "correct": ok,
            })
        return {
            "n": len(sentinels),
            "correct": correct,
            "orientation_ok": correct >= max(1, len(sentinels) - 1),  # tolerate 1 miss
            "probes": probes,
        }
