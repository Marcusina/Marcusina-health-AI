"""
scripts/export_onnx.py
========================
Exports HuggingFace models to ONNX format for fast inference via ONNX Runtime.

Usage:
    python scripts/export_onnx.py              # Export all models
    python scripts/export_onnx.py --model ner
    python scripts/export_onnx.py --model triage
    python scripts/export_onnx.py --model misinfo
    python scripts/export_onnx.py --model sentiment


"""

import os
import sys
import shutil
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from loguru import logger
from app.core.config import get_settings

settings = get_settings()
import warnings
import logging
# Suppress onnxscript version converter warnings — models export correctly at opset 18
logging.getLogger("onnxscript.version_converter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")

# ── Windows safe os.remove patch ─────────────────────────────────────────────
# optimum internally calls os.remove() on temp files that may not exist on
# Windows (different temp path handling). This patch makes it a no-op if
# the file is already gone instead of raising FileNotFoundError.
_original_os_remove = os.remove

def _safe_remove(path):
    try:
        _original_os_remove(path)
    except FileNotFoundError:
        pass  # Already gone — safe to ignore on Windows

os.remove = _safe_remove
# ─────────────────────────────────────────────────────────────────────────────

# NER and misinfo were retired (NER folded into the LLM SOAP call; misinfo
# replaced by the RAG checker), so they are no longer exported or loaded.
MODELS = {
    "triage": {
        "hf_id": settings.HF_TRIAGE_MODEL,
        "task": "text-classification",
    },
    "sentiment": {
        "hf_id": settings.HF_SENTIMENT_MODEL,
        "task": "text-classification",
    },
}


def export_model(name: str, config: dict):
    """
    Export a HuggingFace model to ONNX.

    Uses ORTModel.from_pretrained(..., export=True) with opset=17.
    opset 17 is required because LayerNormalization (used by BERT/RoBERTa)
    has no implementation below opset 17 in the ONNX spec.
    """
    from optimum.onnxruntime import (
        ORTModelForSequenceClassification,
        ORTModelForTokenClassification,
    )

    output_dir = Path(settings.ONNX_MODELS_DIR) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    if onnx_path.exists():
        logger.info(f"[{name}] ONNX model already exists at {onnx_path}. Skipping.")
        return

    logger.info(f"[{name}] Exporting {config['hf_id']} → {output_dir}")
    logger.info(f"[{name}] Task: {config['task']}, opset: 17")

    export_kwargs = dict(
        model_id=config["hf_id"],
        export=True,
              # LayerNormalization requires opset >= 17
    )

    if config["task"] == "token-classification":
        model = ORTModelForTokenClassification.from_pretrained(**export_kwargs)
    else:
        model = ORTModelForSequenceClassification.from_pretrained(**export_kwargs)

    model.save_pretrained(str(output_dir))

    if not onnx_path.exists():
        raise RuntimeError(
            f"[{name}] Export appeared to succeed but {onnx_path} was not found. "
            f"Check the output_dir for what was actually saved."
        )

    size_mb = onnx_path.stat().st_size / 1e6
    logger.info(f"[{name}] Exported successfully. Size: {size_mb:.1f} MB")

    _quantize_int8(name, output_dir, onnx_path)


def _quantize_int8(name: str, output_dir: Path, onnx_path: Path):
    """
    INT8 quantization disabled.

    Root cause: onnxruntime.quantization requires opset <= 15 for static shape
    inference, but torch 2.11 exports at opset 18 (LayerNormalization requires
    opset >= 17). These two constraints are incompatible with the current package
    versions installed.

    Float32 ONNX Runtime is already 4-8x faster than raw PyTorch for inference.
    INT8 would give an additional ~1.5x speedup — pursue it only after deployment
    is stable, by either:
      Option A: pip install onnxruntime==1.15.1 (supports opset 15, older quantizer)
      Option B: Use Optimum's newer QuantizationConfig API when opset 18 support lands
    """
    logger.info(
        f"[{name}] Skipping INT8 quantization (opset 18 / onnxruntime incompatibility). "
        f"Float32 ONNX is production-ready."
    )

def main():
    parser = argparse.ArgumentParser(description="Export HuggingFace models to ONNX")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to export. Default: all",
    )
    args = parser.parse_args()

    os.makedirs(settings.ONNX_MODELS_DIR, exist_ok=True)

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}

    for name, config in targets.items():
        logger.info(f"{'='*50}")
        logger.info(f"Processing: {name} ({config['hf_id']})")
        logger.info(f"{'='*50}")
        export_model(name, config)

    logger.info("All exports complete.")
    logger.info("Next step — start the server:")
    logger.info("  uvicorn app.main:app --reload --port 8001")


if __name__ == "__main__":
    main()