# src/persistence/models.py
"""Cross-repo value objects + business constants + pure status derivation.

WordRef is the unit of practice threading vocab_selector → agent → evaluator.
MASTERY_CAP caps the mastery score so demotion path stays short.
derive_status is a pure function over (current_status, mastery_score, is_target).
"""
from typing import NamedTuple


class WordRef(NamedTuple):
    """A (word, context) pair — the unit of practice."""
    word: str
    context: str = ""


MASTERY_CAP: int = 2


def derive_status(
    current_status: str | None,
    mastery_score: int,
    is_target: bool = False,
) -> str:
    """Derive vocab_stats.status from inputs.

    Returns one of: 'mastered' | 'learning' | 'exposed' | current_status.
    """
    if mastery_score >= MASTERY_CAP:
        return "mastered"
    if current_status == "mastered":
        return "learning" if mastery_score <= 1 else "mastered"
    if is_target:
        return "learning"
    if current_status is None:
        return "exposed"
    return current_status
