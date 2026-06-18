"""
Task definitions — what we evaluate, how, and against which bar.

Each TaskConfig captures everything model-specific so the harness stays generic:
the production model id, the raw→canonical label mapping, the positive class and
threshold, the sentinel probes that guard label orientation, and the go-live
target metric (the number this task must clear before it can ship).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings

settings = get_settings()


@dataclass
class TaskConfig:
    name: str
    model_id: str
    label_map: dict[str, str]
    canonical_labels: list[str]
    positive_label: str | None
    threshold: float | None
    sentinels: list[tuple[str, str]]
    default_dataset: str
    target: str                      # human-readable go-live bar
    target_metric: str = "f1"        # which metric the target refers to
    notes: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Misinfo (health misinformation detection)
# ──────────────────────────────────────────────────────────────────────────────
#
# Current production model: jy46604790/Fake-News-Bert-Detect — a GENERIC
# fake-news classifier trained on news articles, not health claims. Its raw
# labels are LABEL_0 / LABEL_1. Per the model card, LABEL_1 = real/reliable and
# LABEL_0 = fake. We map accordingly; the sentinels below will catch us if that
# orientation is wrong on this checkpoint.

MISINFO = TaskConfig(
    name="misinfo",
    model_id=settings.HF_MISINFO_MODEL,
    label_map={"LABEL_0": "misinfo", "LABEL_1": "reliable"},
    canonical_labels=["misinfo", "reliable"],
    positive_label="misinfo",
    threshold=settings.MISINFO_THRESHOLD,   # production gate (0.75)
    sentinels=[
        ("Bleach injections cure COVID-19.", "misinfo"),
        ("5G networks spread the coronavirus.", "misinfo"),
        ("Regular handwashing reduces the spread of infectious disease.", "reliable"),
        ("Smoking tobacco increases the risk of lung cancer.", "reliable"),
    ],
    default_dataset="pubhealth",
    target="precision >= 0.90 on the misinfo class (avoid flagging true health "
           "info); recall reported but secondary — misinfo moderation favours "
           "precision, with human review on flags.",
    target_metric="precision",
    notes="Generic news model on health-claim domain. Expected to underperform; "
          "this run quantifies the gap to justify claim-detection + retrieval.",
)


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment (patient message tone)
# ──────────────────────────────────────────────────────────────────────────────
#
# cardiffnlp/twitter-roberta-base-sentiment-latest emits negative/neutral/
# positive directly (identity map). Scored on TweetEval/sentiment — the
# benchmark it was trained for, so a fair upper bound (in-domain; Marcusina's
# health-chat domain will differ). 3-class sentiment tops out ~0.70 macro-F1.

SENTIMENT = TaskConfig(
    name="sentiment",
    model_id=settings.HF_SENTIMENT_MODEL,
    label_map={"negative": "negative", "neutral": "neutral", "positive": "positive"},
    canonical_labels=["negative", "neutral", "positive"],
    positive_label=None,             # headline on macro-F1, not one class
    threshold=None,
    sentinels=[
        ("I am so happy with the care I received, thank you!", "positive"),
        ("This is the worst experience, I am furious and disappointed.", "negative"),
        ("The appointment is scheduled for Tuesday at 3pm.", "neutral"),
    ],
    default_dataset="tweeteval_sentiment",
    target="macro-F1 >= 0.65 (realistic ceiling for 3-class sentiment).",
    target_metric="f1",
    notes="Solid general model; not a weak point. Scored to fill the gate table.",
)


TASKS: dict[str, TaskConfig] = {t.name: t for t in (MISINFO, SENTIMENT)}


def get_task(name: str) -> TaskConfig:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[name]
