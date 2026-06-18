"""
CLI entry point for the eval harness.

Examples
--------
  # Score the current misinfo model on the curated offline sample (fast):
  python -m eval.run_eval --task misinfo --dataset sample

  # Score on PUBHEALTH (downloads on first run, then cached):
  python -m eval.run_eval --task misinfo --dataset pubhealth --max 500

  # Measure Whisper WER on your own clips, and A/B base vs large-v3:
  python -m eval.run_eval --task asr --manifest eval/data/asr_clips.jsonl \
      --compare base,large-v3

If the requested dataset can't be loaded (e.g. no network for pubhealth), the
harness falls back to the curated sample so a number is always produced.
"""

from __future__ import annotations

import argparse
import sys

from eval import datasets as ds
from eval.harness import run_classification, run_misinfo_rag
from eval.tasks import get_task, TASKS


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so reports never crash on
    # an unencodable character.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Marcusina model evaluation harness")
    ap.add_argument("--task", default="misinfo", choices=sorted(TASKS) + ["asr", "ner"],
                    help="which task/model to evaluate ('asr' = Whisper WER, "
                         "'ner' = entity F1 on NCBI-disease)")
    ap.add_argument("--dataset", default=None,
                    help=f"dataset name ({ds.available_datasets()}); "
                         "defaults to the task's default dataset")
    ap.add_argument("--model", default="classifier", choices=["classifier", "rag"],
                    help="misinfo only: 'classifier' (old ONNX) or 'rag' (retrieve + LLM judge)")
    ap.add_argument("--max", type=int, default=500,
                    help="max examples (benchmark datasets only)")
    ap.add_argument("--manifest", default=None,
                    help="ASR manifest JSONL ({audio, reference} per line)")
    ap.add_argument("--compare", default=None,
                    help="ASR only: comma-separated Whisper sizes to A/B, "
                         "e.g. base,large-v3")
    ap.add_argument("--no-save", action="store_true", help="don't write a JSON report")
    args = ap.parse_args(argv)

    # ── ASR / Whisper WER ──────────────────────────────────────────────────────
    if args.task == "asr":
        from eval import asr
        if not args.manifest:
            ap.error("--task asr requires --manifest pointing to a JSONL of "
                     "{audio, reference} clips (see eval/asr.py docstring).")
        if args.compare:
            asr.compare(args.manifest, [s.strip() for s in args.compare.split(",")])
        else:
            asr.evaluate(args.manifest)
        return 0

    # ── NER / entity F1 ─────────────────────────────────────────────────────────
    if args.task == "ner":
        from eval import ner
        from app.core.config import get_settings
        dataset = ds.load_ncbi_disease(max_examples=args.max)
        ner.evaluate(dataset, model_id=get_settings().HF_NER_MODEL, save=not args.no_save)
        return 0

    task = get_task(args.task)
    dataset_name = args.dataset or task.default_dataset

    try:
        if dataset_name == "pubhealth":
            dataset = ds.get_dataset("pubhealth", max_examples=args.max)
        else:
            dataset = ds.get_dataset(dataset_name)
    except Exception as e:
        print(f"[warn] could not load dataset '{dataset_name}': {e}\n"
              f"[warn] falling back to curated sample.", file=sys.stderr)
        dataset = ds.get_dataset("sample")

    # RAG misinfo path (retrieve + LLM judge) — one LLM call per example.
    if args.task == "misinfo" and args.model == "rag":
        run_misinfo_rag(dataset, max_examples=args.max if dataset_name == "pubhealth" else None,
                        save=not args.no_save)
        return 0

    run_classification(task, dataset, save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
