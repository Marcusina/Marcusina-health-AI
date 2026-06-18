# Marcusina Evaluation Harness

**Rule of the house:** no model ships without a score, and no score is trusted
until it passes the label-orientation calibration probe.

Public benchmarks here let us **compare candidate models and reject bad ones**.
They do **not** certify go-live — that still requires Marcusina's own labelled
data + clinical sign-off. Treat every number below as a floor on a *proxy*
domain, not a guarantee on patient traffic.

## Layout

| File | Role |
|------|------|
| `metrics.py` | Per-task metrics: classification report (per-class P/R/F1, confusion, positive-class headline), safety recall (triage/distress), WER (Whisper). |
| `datasets.py` | Loaders → normalised `Example{text,label}`. PUBHEALTH (via the Hub's Parquet branch) + curated offline sample. |
| `models.py` | HF classifier adapter. Loads the *same* checkpoint production serves; explicit `label_map`; `calibrate()` guards label orientation. |
| `tasks.py` | Per-task config: model id, label map, threshold, sentinels, **go-live target**. |
| `harness.py` | Runs model + dataset + metric → prints scorecard, saves JSON under `reports/`. |
| `run_eval.py` | CLI. |
| `data/health_claims_sample.jsonl` | 24 curated true/false health claims; always runs offline. |

## Run

```bash
# Fast offline smoke test (curated sample):
python -m eval.run_eval --task misinfo --dataset sample

# Real benchmark (PUBHEALTH, downloads once then cached):
python -m eval.run_eval --task misinfo --dataset pubhealth --max 1200

# Whisper WER on your own clips, A/B base vs large-v3:
python -m eval.run_eval --task asr --manifest eval/data/asr_clips.jsonl \
    --compare base,large-v3
```

The ASR manifest is a JSONL of `{"audio": "clips/x.wav", "reference": "..."}`
(paths relative to the manifest). We have **no checked-in audio benchmark** —
producing a real WER number needs clips with human reference transcripts. Until
then `large-v3` is upgraded on the expectation of lower WER; `compare()` will
quantify it the moment clips exist.

Reports are written to `eval/reports/` (gitignored).

## Why the harness loads the PyTorch checkpoint, not ONNX

Production serves these models via ONNX Runtime, but ONNX is *exported from*
this exact HF checkpoint, so the logits are faithful — and evaluating the
checkpoint directly means we can score a model **before** the ONNX export
pipeline is wired (`scripts/export_onnx.py` is not yet run; `models/` is
gitignored). When ONNX export lands, add an ONNX adapter and diff the two to
confirm export parity.

---

## Baseline result — misinfo model (2026-06-18)

**Model:** `jy46604790/Fake-News-Bert-Detect` (the current production misinfo
model — a *generic news* fake-news classifier, not health-trained).

### PUBHEALTH test, n=1200, threshold=0.75

| class | precision | recall | F1 | support |
|-------|-----------|--------|----|---------|
| **misinfo** (positive) | **0.557** | 0.925 | 0.695 | 615 |
| reliable | 0.742 | 0.226 | 0.346 | 585 |
| accuracy | | | 0.584 | 1200 |
| macro-F1 | | | 0.521 | |

It predicted **misinfo for 1022 / 1200** inputs.

### Curated sample, n=24 (balanced)

Predicted **misinfo for all 24** → precision 0.50 (chance), single-class
collapse flagged by the harness.

### Verdict: **FAIL** (bar: misinfo precision ≥ 0.90)

The model is barely above the base rate. The decisive failure for a healthcare
product chasing "no false positives": on genuinely **true** health claims it is
right only **22.6%** of the time — it flags **77%** of accurate health
information as misinformation (453 of 585 true claims). Calibration sentinels
caught this immediately (it flagged "regular handwashing reduces infection
spread" and "smoking increases lung cancer risk" as misinfo).

This confirms the prior call: **misinfo is not fixable by a model swap.** A
PUBHEALTH-fine-tuned classifier would beat 0.557 but won't reach a defensible
precision bar on open health claims. The real path is **claim-detection +
retrieval against trusted sources (WHO/CDC/PubMed)** with human review on flags.

## RAG misinfo vs the classifier (2026-06-18)

The retrieval-grounded checker (`app/rag`, `granite4.1:8b` judge over a 35-doc
WHO/CDC/NHS/NIH seed corpus), scored via `python -m eval.run_eval --task misinfo
--model rag`. Mapping: `contradicted → misinfo`; `supported / unsupported /
not_health_claim → reliable` (precision-oriented — only flag what evidence
*contradicts*).

| Dataset | RAG | Prior classifier |
|---------|-----|------------------|
| Curated sample (n=24, balanced) | **P 1.00 / R 1.00 / F1 1.00** | P 0.50 (chance, all-misinfo collapse) |
| PUBHEALTH (n=40 subset) | P — / R 0.00 (all 40 → `unsupported`) | P 0.557 / R 0.925 |

**The two systems fail in opposite directions — and that's the whole point:**

- On topics the corpus **covers** (the curated common-misinfo set), RAG is
  perfect: every false claim `contradicted`, every true claim `supported` and
  *not* flagged. The classifier scored chance on the same set.
- On PUBHEALTH, the 35-doc seed corpus covers **none** of its niche fact-checks,
  so RAG returns `unsupported` for all 40 and **abstains** (recall 0, but **zero
  false positives**). The classifier instead flags almost everything (recall
  0.925) at the cost of crying wolf on true info (precision 0.557).

For a healthcare advisory product, RAG's failure mode (**abstain when no
evidence**) is far safer than the classifier's (**false-flag true health info**) —
and it is *fixable by growing the corpus*, whereas the classifier's lack of
factual grounding is not. **The next lever is corpus ingestion** (real
PubMed/WHO/CDC snapshots), not the model. `verdict_spread` in each report shows
coverage: PUBHEALTH was `{unsupported: 40}`, curated was `{contradicted: 12,
supported: 9, unsupported: 3}`.

## Baseline result — sentiment model (2026-06-18)

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`. **Benchmark:**
TweetEval/sentiment test, n=1000 (the dataset this model was trained for — fair
in-domain, but Marcusina's health-chat domain will differ).

| | macro-F1 | accuracy |
|---|----------|----------|
| **0.703** | | 0.698 |

Per-class F1: negative 0.699, neutral 0.685, positive 0.726. Calibration 3/3.
**Verdict: PASS** (bar: macro-F1 ≥ 0.65). This is at the realistic ceiling for
3-class sentiment — not a weak point. Re-score on real patient messages before
trusting it in-domain.

## Baseline result — NER model (2026-06-18)

**Model:** `d4data/biomedical-ner-all` (MACCROBAT schema, 84 types).
**Benchmark:** NCBI-disease test, n=800. We map the model's `Disease_disorder`
type onto NCBI's single `Disease` type and score entity spans.

| match mode | precision | recall | F1 |
|------------|-----------|--------|-----|
| exact span | 0.498 | 0.286 | **0.363** |
| relaxed (overlap) | 0.774 | 0.444 | 0.564 |

**Verdict: FAIL** (bar: exact entity F1 ≥ 0.80).

Reading it honestly: relaxed **precision 0.77** vs exact 0.50 shows that boundary
/ annotation-guideline differences (cross-dataset) inflate the strict
false-positive count — when the model flags a disease it usually overlaps a real
one. But **recall is only 0.44 even relaxed** — it misses over half of NCBI's
disease mentions regardless of boundary tolerance. That is a genuine weakness,
not just schema mismatch.

⚠️ Correction to a prior assumption: the frequently-cited "~84% F1" is d4data's
**in-domain (MACCROBAT)** score and does **not** transfer to disease NER. On a
disease benchmark this model is weak on recall. If SOAP/ICD extraction depends on
catching disease mentions, NER needs revisiting (scispaCy `en_ner_bc5cdr_md`, or
a model fine-tuned on NCBI/BC5CDR) — it is not "fine as-is".

## Per-task go-live gate (proposed)

"95% accuracy on all models" is not a shippable target — the metric differs per
task. Concrete bars this harness checks against:

| Task | Metric that matters | Proposed gate | Status |
|------|--------------------|---------------|--------|
| Misinfo | precision on `misinfo` (human reviews flags) | ≥ 0.90 | ❌ 0.557 (PUBHEALTH) |
| Triage (emergency) | **recall/sensitivity** on emergencies | ≥ 0.99 via rules+human | rules net in place; ML advisory only |
| ASR (Whisper) | WER on medical speech | report + improve vs `base` | upgraded `base`→`large-v3` (2026-06-18); WER pending audio clips |
| NER | entity-level F1 (exact) | ≥ 0.80 | ❌ 0.363 exact / 0.564 relaxed (NCBI-disease) |
| Sentiment | macro-F1 | ≥ 0.65 (realistic 3-class ceiling) | ✅ 0.703 (TweetEval) |

None of these certify go-live on their own — they gate *candidate selection*.
Clinical validation on Marcusina's own labelled data remains mandatory.
