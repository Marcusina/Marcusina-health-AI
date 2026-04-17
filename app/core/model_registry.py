"""
Model Registry (ONNX Runtime)
==============================
All models run through ONNX Runtime — no PyTorch at inference time.
- 4-8× faster inference than PyTorch
- Much smaller memory footprint
- No CUDA dependency (CPU inference is production-grade)
- Models are loaded once per worker process and kept in memory

faster-whisper uses CTranslate2 backend (INT8 quantized):
- No pkg_resources issues
- No system ffmpeg dependency (uses libav via PyAV)
- 4× faster than openai-whisper on CPU

ONNX models are exported once via: python scripts/export_onnx.py
"""

from __future__ import annotations
import os
import json
import asyncio
from pathlib import Path
from loguru import logger
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
import onnxruntime as ort
import faiss
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


def _make_ort_session(model_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime session with optimised thread settings."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = settings.ONNX_INTRA_THREADS
    opts.inter_op_num_threads = settings.ONNX_INTER_THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]  # Add CUDAExecutionProvider first if GPU available
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


class ModelRegistry:
    """
    Singleton registry — one instance per worker process.
    All heavy objects loaded once at startup; zero cost per request.
    """

    def __init__(self):
        self._whisper: Optional[WhisperModel] = None
        self._ner_session: Optional[ort.InferenceSession] = None
        self._ner_tokenizer = None
        self._triage_session: Optional[ort.InferenceSession] = None
        self._triage_tokenizer = None
        self._misinfo_session: Optional[ort.InferenceSession] = None
        self._misinfo_tokenizer = None
        self._sentiment_session: Optional[ort.InferenceSession] = None
        self._sentiment_tokenizer = None
        self._embedder: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._faiss_metadata: Optional[list] = None
        self._ready: bool = False

    def load_all(self):
        """
        Called once at startup per worker process.
        Downloads if not cached, loads into memory.
        """
        logger.info(f"[Worker PID {os.getpid()}] Loading AI models...")

        self._load_whisper()
        self._load_onnx_models()
        self._load_embedder()
        self._load_faiss()

        self._ready = True
        logger.info(f"[Worker PID {os.getpid()}] All models ready.")

    # ── faster-whisper ───────────────────────────────────────────────────────

    def _load_whisper(self):
        logger.info(f"Loading faster-whisper [{settings.WHISPER_MODEL_SIZE}] "
                    f"compute={settings.WHISPER_COMPUTE_TYPE} device={settings.WHISPER_DEVICE}")
        self._whisper = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            num_workers=settings.WHISPER_NUM_WORKERS,
            download_root=os.path.join(settings.MODELS_DIR, "whisper"),
        )
        logger.info("faster-whisper loaded.")

    # ── ONNX NLP models ──────────────────────────────────────────────────────

    def _load_onnx_models(self):
        onnx_dir = Path(settings.ONNX_MODELS_DIR)

        models = [
            ("ner",       settings.HF_NER_MODEL,       "_ner_session",       "_ner_tokenizer"),
            ("triage",    settings.HF_TRIAGE_MODEL,     "_triage_session",    "_triage_tokenizer"),
            ("misinfo",   settings.HF_MISINFO_MODEL,    "_misinfo_session",   "_misinfo_tokenizer"),
            ("sentiment", settings.HF_SENTIMENT_MODEL,  "_sentiment_session", "_sentiment_tokenizer"),
        ]

        for name, hf_model_id, session_attr, tokenizer_attr in models:
            onnx_path = onnx_dir / name / "model.onnx"
            if not onnx_path.exists():
                logger.warning(
                    f"ONNX model not found at {onnx_path}. "
                    f"Run: python scripts/export_onnx.py --model {name}\n"
                    f"Falling back to tokenizer-only mode (limited functionality)."
                )
                setattr(self, session_attr, None)
            else:
                logger.info(f"Loading ONNX [{name}] from {onnx_path}")
                setattr(self, session_attr, _make_ort_session(str(onnx_path)))

            # Tokenizer is always loaded from HuggingFace cache (fast, ~10MB)
            logger.info(f"Loading tokenizer for [{name}]...")
            setattr(self, tokenizer_attr,
                    AutoTokenizer.from_pretrained(hf_model_id))

    # ── Sentence embedder ─────────────────────────────────────────────────────

    def _load_embedder(self):
        logger.info(f"Loading sentence embedder [{settings.HF_EMBEDDING_MODEL}]...")
        self._embedder = SentenceTransformer(
            settings.HF_EMBEDDING_MODEL,
            cache_folder=os.path.join(settings.MODELS_DIR, "embedders"),
        )
        logger.info("Embedder loaded.")

    # ── FAISS vector index ────────────────────────────────────────────────────

    def _load_faiss(self):
        index_path = settings.FAISS_INDEX_PATH
        meta_path = settings.FAISS_METADATA_PATH

        if os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}...")
            self._faiss_index = faiss.read_index(index_path)
            with open(meta_path) as f:
                self._faiss_metadata = json.load(f)
            logger.info(f"FAISS index loaded: {self._faiss_index.ntotal} vectors.")
        else:
            logger.warning(
                f"FAISS index not found at {index_path}. "
                f"Recommendations will use fallback mode. "
                f"Build it with: python scripts/build_faiss_index.py"
            )
            self._faiss_index = None
            self._faiss_metadata = []

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def whisper(self) -> WhisperModel:
        self._assert_ready("whisper")
        return self._whisper

    @property
    def ner(self) -> tuple[Optional[ort.InferenceSession], object]:
        return self._ner_session, self._ner_tokenizer

    @property
    def triage(self) -> tuple[Optional[ort.InferenceSession], object]:
        return self._triage_session, self._triage_tokenizer

    @property
    def misinfo(self) -> tuple[Optional[ort.InferenceSession], object]:
        return self._misinfo_session, self._misinfo_tokenizer

    @property
    def sentiment(self) -> tuple[Optional[ort.InferenceSession], object]:
        return self._sentiment_session, self._sentiment_tokenizer

    @property
    def embedder(self) -> SentenceTransformer:
        self._assert_ready("embedder")
        return self._embedder

    @property
    def faiss_index(self) -> Optional[faiss.Index]:
        return self._faiss_index

    @property
    def faiss_metadata(self) -> list:
        return self._faiss_metadata or []

    def _assert_ready(self, name: str):
        if not self._ready:
            raise RuntimeError(f"Model registry not initialised. {name} not available.")


# Global singleton — one per worker process
_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return _registry


# ── ONNX inference helpers ────────────────────────────────────────────────────

def run_onnx_classifier(
    session: ort.InferenceSession,
    tokenizer,
    text: str,
    max_length: int = 512,
) -> list[dict]:
    """
    Run a HuggingFace sequence-classification ONNX model.
    Returns list of {label, score} dicts, sorted by score descending.
    """
    import numpy as np
    from scipy.special import softmax

    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    # ONNX Runtime expects numpy arrays
    ort_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in session.get_inputs()]}
    logits = session.run(None, ort_inputs)[0][0]
    scores = softmax(logits).tolist()

    id2label = tokenizer.config.id2label if hasattr(tokenizer, "config") else {i: str(i) for i in range(len(scores))}
    return sorted(
        [{"label": id2label.get(i, str(i)), "score": float(s)} for i, s in enumerate(scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
