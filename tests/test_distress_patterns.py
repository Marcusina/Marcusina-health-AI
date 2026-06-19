"""
Regression guard for the distress-pattern config (config/distress_patterns.json).

v2.0 was bloated with ~365 bare single-word alternatives (today, now, done, care,
love, over, alone, kids, …) that fired on ordinary benign messages — every false
match feeds the fail-safe escalation path, so this was a real safety/UX bug.
v2.1 removed the noise while keeping the multi-word ideation phrases.

These tests run against the REAL config via the production matcher, so they fail
if the bare-word noise ever creeps back in (benign) or if real ideation phrasing
stops matching (recall).
"""

from __future__ import annotations

import pytest

from app.safety.rules import match_distress


# Everyday telehealth/social messages that must NOT be flagged as distress.
BENIGN = [
    "I'm feeling much better today, thanks for asking!",
    "Can we reschedule my appointment to later this week?",
    "I've been sleeping well and my energy is back.",
    "Just finished my workout, feeling great.",
    "I'm done with my medication course now.",
    "Thanks so much, I really appreciate the care I received.",
    "I love the new diet plan, it's working.",
    "I'm a bit tired after work but otherwise fine.",
    "My back pain is almost gone now.",
    "I'll be over at the clinic at noon.",
    "Feeling alone in a new city but excited to explore.",
    "The kids are driving me crazy today, lol.",
    "I'm so done with this cold weather!",
    "Work has been killing me this week, so busy.",
    "I could sleep forever after that long shift.",
    "I took my pills this morning as prescribed.",
    "I could die for a cup of coffee right now, haha.",
    "I miss my family back home but we video call weekly.",
    "Please update my address on file.",
    "I quit smoking last month and feel amazing.",
]

# Genuine ideation / self-harm phrasing that MUST still be caught (high recall).
TRUE_DISTRESS = [
    "I want to kill myself.",
    "I don't want to be alive anymore.",
    "I've been thinking about ending my life.",
    "Everyone would be better off without me.",
    "I've been cutting myself to cope.",
    "I have a plan to overdose on my pills tonight.",
    "I feel hopeless and trapped with no way out.",
    "Nothing matters anymore and I can't go on.",
    "I wish I were dead.",
    "I'm planning to hurt myself.",
    "I feel numb and empty inside, dead inside.",
    "I've been having suicidal thoughts.",
    "Better off dead than living like this.",
    "I just want to go to sleep and not wake up.",
]


@pytest.mark.parametrize("text", BENIGN)
def test_benign_message_does_not_flag_distress(text):
    assert match_distress(text) == [], f"benign text falsely flagged: {match_distress(text)}"


@pytest.mark.parametrize("text", TRUE_DISTRESS)
def test_real_ideation_still_flags(text):
    assert match_distress(text), "real ideation phrasing was not caught"


def test_bare_common_words_are_not_standalone_triggers():
    # The exact noise that defined the v2.0 bug — none may flag on their own.
    for word in ["today", "now", "done", "care", "love", "over", "alone",
                 "kids", "sleep", "finished", "gone", "tired", "plan"]:
        assert match_distress(word) == [], f"bare word still triggers distress: {word!r}"
