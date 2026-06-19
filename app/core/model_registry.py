

from __future__ import annotations
import os
import json
import asyncio
from pathlib import Path
from loguru import logger
from typing import Optional

import numpy as np
from scipy.special import softmax
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
import onnxruntime as ort
import faiss
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


def _make_ort_session(model_path: str) -> "ort.InferenceSession":
    """Create an ONNX Runtime session with optimised thread settings."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = settings.ONNX_INTRA_THREADS
    opts.inter_op_num_threads = settings.ONNX_INTER_THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)

    # Store path on session so run_onnx_classifier can find config.json
    session._model_path = model_path
    session._valid_input_names = frozenset(i.name for i in session.get_inputs())
    return session

class ModelRegistry:
    """
    Singleton registry — one instance per worker process.
    All heavy objects loaded once at startup; zero cost per request.
    """

    def __init__(self):
        self._whisper: Optional[WhisperModel] = None
        # NER + misinfo ONNX models were retired: NER (d4data) failed eval (entity
        # F1 0.36) and is now folded into the LLM SOAP call; the misinfo classifier
        # (0.557 precision) was replaced by the RAG checker (app/rag). Neither is
        # loaded anymore — see app/clinical and app/rag.
        self._triage_session: Optional[ort.InferenceSession] = None
        self._triage_tokenizer = None
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
        import json
        onnx_dir = Path(settings.ONNX_MODELS_DIR)

        models = [
            ("triage",    settings.HF_TRIAGE_MODEL,     "_triage_session",    "_triage_tokenizer"),
            ("sentiment", settings.HF_SENTIMENT_MODEL,  "_sentiment_session", "_sentiment_tokenizer"),
        ]

        # Store id2label maps separately — keyed by model name
        self._id2label: dict[str, dict] = {}

        for name, hf_model_id, session_attr, tokenizer_attr in models:
            onnx_path = onnx_dir / name / "model.onnx"
            config_path = onnx_dir / name / "config.json"

            # Load id2label from the saved config.json
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                raw = cfg.get("id2label", {})
                # Keys from JSON are always strings — convert to int keys
                self._id2label[name] = {int(k): v for k, v in raw.items()} if raw else {}
                if self._id2label[name]:
                    logger.info(f"[{name}] Labels: {self._id2label[name]}")
                else:
                    logger.warning(f"[{name}] id2label is empty in config.json — check verify_labels.py")
            else:
                self._id2label[name] = {}
                logger.warning(f"[{name}] No config.json found at {config_path}")

            if not onnx_path.exists():
                logger.warning(f"ONNX model not found: {onnx_path}. Run: python scripts/export_onnx.py --model {name}")
                setattr(self, session_attr, None)
            else:
                logger.info(f"Loading ONNX [{name}]...")
                setattr(self, session_attr, _make_ort_session(str(onnx_path)))

            logger.info(f"Loading tokenizer [{name}] from {hf_model_id}...")
            setattr(self, tokenizer_attr, AutoTokenizer.from_pretrained(hf_model_id))

    def get_id2label(self, name: str) -> dict:
        return self._id2label.get(name, {})
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
    def triage(self) -> tuple[Optional[ort.InferenceSession], object]:
        return self._triage_session, self._triage_tokenizer

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


def run_onnx_classifier(
    session: "ort.InferenceSession",
    tokenizer,
    text: str,
    max_length: int = 512,
    id2label: dict = None,
) -> list[dict]:
    """
    Run a HuggingFace sequence-classification ONNX model.
    Returns list of {label, score} sorted by score descending.

    id2label must be passed in explicitly from the registry loader —
    do not rely on session._model_path which is not guaranteed to be set.
    """
    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=max_length,
        truncation=True,
        padding=True,
    )

    valid_names = getattr(session, "_valid_input_names", frozenset(i.name for i in session.get_inputs()))
    ort_inputs = {
    k: v.astype("int64") if v.dtype.kind == "i" else v
    for k, v in inputs.items()
    if k in valid_names
}

    logits = session.run(None, ort_inputs)[0][0]
    scores = softmax(logits).tolist()

    if not id2label:
        id2label = {i: str(i) for i in range(len(scores))}

    return sorted(
        [{"label": id2label.get(i, str(i)), "score": float(s)}
         for i, s in enumerate(scores)],
        key=lambda x: x["score"],
        reverse=True,
    )