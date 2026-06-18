"""
Clinical generation lane (local LLM).

Generative, post-consultation, **documentation-assist** tasks — always reviewed
by a clinician, never autonomous (docs/AI-PLATFORM-DESIGN.md §6):

  * generate_soap(transcript)    -> SOAP note + extracted entities + ICD suggestions
  * generate_summary(transcript) -> patient-friendly visit summary

Design choices:
  * Entity extraction is folded into the SOAP LLM call (the previous standalone
    ONNX NER scored 0.36 F1 on NCBI-disease; the LLM has full transcript context).
  * ICD codes are mapped deterministically from config/icd_map.json — the LLM
    extracts diagnoses, but codes are never invented by the model.
  * Faithfulness: the model is instructed to use only what's in the transcript.
  * Graceful degradation: if the LLM is unavailable, a clearly-marked
    (`degraded: true`) rule-based draft is returned rather than nothing.
"""

from app.clinical.soap import generate_soap
from app.clinical.summary import generate_summary

__all__ = ["generate_soap", "generate_summary"]
