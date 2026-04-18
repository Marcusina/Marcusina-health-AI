"""
scripts/verify_labels.py
Run: python scripts/verify_labels.py
"""
import json, os
from pathlib import Path

models = ["ner", "triage", "misinfo", "sentiment"]

for name in models:
    config_path = Path(f"models/onnx/{name}/config.json")
    if not config_path.exists():
        print(f"[{name}] MISSING config.json — re-export this model")
        continue

    cfg = json.loads(config_path.read_text())
    id2label = cfg.get("id2label", {})
    num_labels = cfg.get("num_labels", "?")

    if not id2label:
        print(f"[{name}] WARNING — id2label is EMPTY. Labels will be '0','1' etc. May cause false positives.")
    else:
        print(f"[{name}] OK — {num_labels} labels: {id2label}")