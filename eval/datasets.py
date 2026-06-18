"""
Dataset loaders — normalise public benchmarks (and local samples) into a single
Example schema the harness can score.

Every loader returns list[Example] with a canonical `label` already mapped into
the task's label space, so the harness/metrics never need to know dataset
quirks. Loaders degrade gracefully: if a benchmark can't be reached, the caller
can fall back to the checked-in curated sample so the loop still runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class Example:
    text: str
    label: str                       # canonical task label
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    task: str                        # "misinfo" | "ner" | "triage" | ...
    labels: list[str]                # canonical label space, fixed order
    examples: list[Example]

    def __len__(self) -> int:
        return len(self.examples)

    def texts(self) -> list[str]:
        return [e.text for e in self.examples]

    def gold(self) -> list[str]:
        return [e.label for e in self.examples]


# ──────────────────────────────────────────────────────────────────────────────
# Local JSONL
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path, task: str, labels: list[str]) -> Dataset:
    path = Path(path)
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(Example(text=row["text"], label=row["label"],
                                    meta={k: v for k, v in row.items()
                                          if k not in ("text", "label")}))
    return Dataset(name=path.stem, task=task, labels=labels, examples=examples)


def load_curated_health_claims() -> Dataset:
    """Checked-in misinfo sample — always available, runs offline."""
    return load_jsonl(DATA_DIR / "health_claims_sample.jsonl",
                      task="misinfo", labels=["misinfo", "reliable"])


# ──────────────────────────────────────────────────────────────────────────────
# PUBHEALTH (health_fact) — public health misinformation benchmark
# ──────────────────────────────────────────────────────────────────────────────

# PUBHEALTH original 4-class veracity labels → our binary misinfo task.
# "true" is the only reliable class; everything else is treated as misinfo
# (false / partially-false / unverifiable claims should all be flagged).
_PUBHEALTH_MAP = {
    "true": "reliable",
    "false": "misinfo",
    "mixture": "misinfo",
    "unproven": "misinfo",
}


# PUBHEALTH's ClassLabel integer encoding (health_fact original order).
_PUBHEALTH_INT2STR = {0: "false", 1: "mixture", 2: "true", 3: "unproven"}


def load_pubhealth(split: str = "test", max_examples: int | None = 500,
                   binary: bool = True) -> Dataset:
    """
    Load PUBHEALTH from HuggingFace. The original `health_fact` dataset is
    script-based, which `datasets>=3` no longer runs, so we read directly from
    the Hub's auto-converted Parquet branch (refs/convert/parquet) with pandas.
    Cached by huggingface_hub after the first call.

    The claim text is used as the model input (matching how Marcusina sees a
    short health claim, not a full article). The 4-class veracity label is
    mapped to binary misinfo/reliable.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="health_fact", repo_type="dataset",
        revision="refs/convert/parquet",
        filename=f"default/{split}/0000.parquet",
    )
    df = pd.read_parquet(path, columns=["claim", "label"])

    examples: list[Example] = []
    for claim, label in zip(df["claim"], df["label"]):
        raw = label
        if isinstance(raw, (int,)) or (hasattr(raw, "item")):
            raw = _PUBHEALTH_INT2STR.get(int(raw), "")
        raw = str(raw).strip().lower()
        if raw not in _PUBHEALTH_MAP:
            continue  # drop the rare "-1" / unlabelled rows
        claim = (claim or "").strip()
        if not claim:
            continue
        canon = _PUBHEALTH_MAP[raw] if binary else raw
        examples.append(Example(text=claim, label=canon, meta={"orig_label": raw}))
        if max_examples and len(examples) >= max_examples:
            break

    labels = ["misinfo", "reliable"] if binary else list(_PUBHEALTH_MAP.keys())
    return Dataset(name=f"pubhealth-{split}", task="misinfo",
                   labels=labels, examples=examples)


# ──────────────────────────────────────────────────────────────────────────────
# TweetEval sentiment — the benchmark cardiffnlp's model was trained for
# ──────────────────────────────────────────────────────────────────────────────

_TWEETEVAL_SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}


def load_tweeteval_sentiment(split: str = "test",
                             max_examples: int | None = 1000) -> Dataset:
    """
    TweetEval / sentiment via the Hub Parquet branch. This is the canonical
    benchmark for cardiffnlp/twitter-roberta-base-sentiment-latest, so it's a
    fair in-domain score (note: in-domain ≠ Marcusina's health chat domain).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="tweet_eval", repo_type="dataset",
        revision="refs/convert/parquet",
        filename=f"sentiment/{split}/0000.parquet",
    )
    df = pd.read_parquet(path)
    examples = []
    for text, label in zip(df["text"], df["label"]):
        canon = _TWEETEVAL_SENTIMENT.get(int(label))
        if canon is None:
            continue
        examples.append(Example(text=str(text), label=canon))
        if max_examples and len(examples) >= max_examples:
            break
    return Dataset(name=f"tweeteval-sentiment-{split}", task="sentiment",
                   labels=["negative", "neutral", "positive"], examples=examples)


# ──────────────────────────────────────────────────────────────────────────────
# NCBI-disease — biomedical NER benchmark (token-level)
# ──────────────────────────────────────────────────────────────────────────────

# NCBI ner_tags ClassLabel: 0=O, 1=B-Disease, 2=I-Disease
_NCBI_TAGS = {0: "O", 1: "B-Disease", 2: "I-Disease"}


@dataclass
class TokenExample:
    tokens: list[str]
    tags: list[str]                  # gold BIO tags (canonical type, e.g. Disease)


@dataclass
class TokenDataset:
    name: str
    task: str
    examples: list["TokenExample"]

    def __len__(self) -> int:
        return len(self.examples)


def load_ncbi_disease(split: str = "test",
                      max_examples: int | None = 800) -> TokenDataset:
    """NCBI-disease via the Hub Parquet branch → tokens + canonical BIO tags."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="ncbi_disease", repo_type="dataset",
        revision="refs/convert/parquet",
        filename=f"ncbi_disease/{split}/0000.parquet",
    )
    df = pd.read_parquet(path, columns=["tokens", "ner_tags"])
    examples = []
    for tokens, tags in zip(df["tokens"], df["ner_tags"]):
        toks = [str(t) for t in tokens]
        bio = [_NCBI_TAGS.get(int(t), "O") for t in tags]
        if not toks:
            continue
        examples.append(TokenExample(tokens=toks, tags=bio))
        if max_examples and len(examples) >= max_examples:
            break
    return TokenDataset(name=f"ncbi-disease-{split}", task="ner", examples=examples)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

# name -> zero-arg (or default-arg) loader
_REGISTRY: dict[str, Callable[..., Dataset]] = {
    "sample": load_curated_health_claims,
    "curated": load_curated_health_claims,
    "pubhealth": load_pubhealth,
    "tweeteval_sentiment": load_tweeteval_sentiment,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)
