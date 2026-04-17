"""
scripts/export_onnx.py
========================
Converts HuggingFace models to ONNX format for fast inference.
Run once during setup — exported models are reused every startup.

Usage:
    python scripts/export_onnx.py              # Export all models
    python scripts/export_onnx.py --model ner  # Export specific model

ONNX Runtime is 4-8x faster than PyTorch for inference:
- No gradient computation overhead
- Graph-level optimizations (operator fusion, constant folding)
- INT8 quantization support
- No CUDA required for production-grade CPU inference
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from loguru import logger
from optimum.exporters.onnx import main_export
from app.core.config import get_settings

settings = get_settings()


MODELS = {
    "ner": {
        "hf_id": settings.HF_NER_MODEL,
        "task": "token-classification",
    },
    "triage": {
        "hf_id": settings.HF_TRIAGE_MODEL,
        "task": "text-classification",
    },
    "misinfo": {
        "hf_id": settings.HF_MISINFO_MODEL,
        "task": "text-classification",
    },
    "sentiment": {
        "hf_id": settings.HF_SENTIMENT_MODEL,
        "task": "text-classification",
    },
}


def export_model(name: str, config: dict):
    """Export a single HuggingFace model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForTokenClassification

    output_dir = Path(settings.ONNX_MODELS_DIR) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    if onnx_path.exists():
        logger.info(f"[{name}] ONNX model already exists at {onnx_path}. Skipping.")
        return

    logger.info(f"[{name}] Exporting {config['hf_id']} → {output_dir}")

    import os

    # Patch: prevent crash when file doesn't exist
    _original_remove = os.remove

    def safe_remove(path):
        if os.path.exists(path):
            _original_remove(path)

    os.remove = safe_remove

    main_export(
        model_name_or_path=config["hf_id"],
        output=output_dir,
        task=config["task"],
        opset=18,  #  critical fix
        use_external_data_format=False,
    )
    onnx_path = output_dir / "model.onnx"
    if not onnx_path.exists():
        raise RuntimeError(f"Export failed: {onnx_path} not found")
    logger.info(f"[{name}] Exported successfully. Size: {onnx_path.stat().st_size / 1e6:.1f} MB")

    # Optional: quantise to INT8 for even faster CPU inference
    _quantize_int8(name, output_dir, onnx_path)


def _quantize_int8(name: str, output_dir: Path, onnx_path: Path):
    """Quantise the ONNX model to INT8 for ~2x faster CPU inference."""
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantized_path = output_dir / "model_int8.onnx"
        if quantized_path.exists():
            logger.info(f"[{name}] INT8 model already exists.")
            return

        logger.info(f"[{name}] Quantising to INT8...")
        quantizer = ORTQuantizer.from_pretrained(str(output_dir))
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=str(output_dir), quantization_config=qconfig)

        # Rename to model.onnx so the registry picks it up
        import shutil


        if onnx_path.exists():
            os.remove(onnx_path)

        shutil.move(str(quantized_path), str(onnx_path))
        logger.info(f"[{name}] INT8 quantisation complete.")
    except Exception as e:
        logger.warning(f"[{name}] INT8 quantisation failed (non-critical): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    os.makedirs(settings.ONNX_MODELS_DIR, exist_ok=True)

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for name, config in targets.items():
        export_model(name, config)

    logger.info("Export complete. Start the server:")
    logger.info("  gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --workers 8 --bind 0.0.0.0:8001")


if __name__ == "__main__":
    main()
