"""
ASR (Whisper) evaluation — Word Error Rate.

Unlike the classification tasks, we have no checked-in audio benchmark (this ties
back to "no labelled data yet"). So this module evaluates against a *manifest*
the user provides: a JSONL where each line is

    {"audio": "clips/visit01.wav", "reference": "the doctor said ..."}

`reference` is the ground-truth human transcript. Paths are relative to the
manifest file. It transcribes each clip with faster-whisper using the project's
config and reports corpus WER (eval.metrics.corpus_wer).

`compare()` runs two model sizes over the same clips so the large-v3 upgrade can
be quantified (Δ WER) instead of assumed — the same "no model trusted without a
score" rule applied to the STT swap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from eval import metrics


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    base = path.parent
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["audio"] = str((base / row["audio"]).resolve())
            rows.append(row)
    return rows


def _transcribe_all(model, clips: list[dict], beam_size: int) -> list[str]:
    hyps = []
    for clip in clips:
        segments, _info = model.transcribe(clip["audio"], beam_size=beam_size,
                                            vad_filter=True)
        hyps.append(" ".join(s.text.strip() for s in segments))
    return hyps


def evaluate(manifest: str | Path, model_size: str | None = None) -> dict:
    """Transcribe the manifest with one Whisper size and report WER."""
    from faster_whisper import WhisperModel
    from app.core.config import get_settings
    settings = get_settings()

    size = model_size or settings.WHISPER_MODEL_SIZE
    clips = load_manifest(manifest)
    print(f"\nASR eval: model={size}  clips={len(clips)}  "
          f"compute={settings.WHISPER_COMPUTE_TYPE} device={settings.WHISPER_DEVICE}")

    t0 = time.perf_counter()
    model = WhisperModel(size, device=settings.WHISPER_DEVICE,
                         compute_type=settings.WHISPER_COMPUTE_TYPE,
                         download_root=f"{settings.MODELS_DIR}/whisper")
    hyps = _transcribe_all(model, clips, settings.WHISPER_BEAM_SIZE)
    refs = [c["reference"] for c in clips]
    res = metrics.corpus_wer(refs, hyps)
    res["model_size"] = size
    res["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  WER mean={res['wer_mean']:.3f}  median={res['wer_median']:.3f}  "
          f"({res['seconds']}s)")
    return res


def compare(manifest: str | Path, sizes: list[str]) -> dict:
    """Run several Whisper sizes over the same clips and report the Δ WER."""
    results = {s: evaluate(manifest, model_size=s) for s in sizes}
    baseline = results[sizes[0]]["wer_mean"]
    print(f"\nWER comparison (baseline = {sizes[0]}):")
    for s in sizes:
        wm = results[s]["wer_mean"]
        delta = wm - baseline
        print(f"  {s:<10} WER={wm:.3f}  Δ={delta:+.3f} vs {sizes[0]}")
    return {"baseline": sizes[0], "results": results}
