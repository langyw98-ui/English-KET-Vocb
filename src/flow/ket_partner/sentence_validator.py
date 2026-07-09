import json
import re
from os.path import dirname, join

from pydantic import BaseModel, Field

_FUNCTION_WORDS_PATH = join(dirname(__file__), "data", "function_words.json")
_LEMMAS_PATH = join(dirname(__file__), "data", "lemmas.json")

with open(_FUNCTION_WORDS_PATH, "r", encoding="utf-8") as f:
    _FUNCTION_WORDS = set(json.load(f))

with open(_LEMMAS_PATH, "r", encoding="utf-8") as f:
    _LEMMAS = json.load(f)


class ValidationResult(BaseModel):
    ok: bool
    words_used: list = Field(default_factory=list)
    non_ket_words: list = Field(default_factory=list)


def _tokenize(sentence: str) -> list:
    return re.findall(r"[A-Za-z']+", sentence)


def _is_proper_noun(token: str) -> bool:
    return token[:1].isupper()


def _candidate_roots(token: str) -> list:
    """Return possible lemma candidates for an English token, handling
    common inflections (verb tense, noun plural, comparatives).

    Strategy: the lowercased token itself is always tried first — many words
    (the, is, cat) are in the KET vocabulary directly. If that doesn't match,
    fall back to: explicit _LEMMAS entries (irregulars like went→go), then
    rule-based suffix stripping (wears→wear, cats→cat). The validator asks
    `get_ket_word_any_context` for each candidate in order — first match
    wins, which prevents over-stemming (e.g., "bins" → tries "bin" only if
    "bin" is actually a known word).
    """
    lower = token.lower()
    candidates = [lower]
    if lower in _LEMMAS:
        candidates.append(_LEMMAS[lower])
    # Plurals / 3rd-person singular. Try BOTH "-es" and "-s" stems: a word
    # like "makes" ends in "es" but its lemma is "make" (verb+s), not "mak".
    # The -es branch handles true -es plurals (watches→watch, boxes→box);
    # the -s branch catches the verb+s case (makes→make, likes→like).
    # get_ket_word tries candidates in order, so an incorrect "mak" candidate
    # is harmless as long as "make" is also in the list.
    if lower.endswith("ies") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")   # stories → story
    if lower.endswith("es") and len(lower) > 3:
        candidates.append(lower[:-2])         # watches → watch, boxes → box
    if lower.endswith("s") and len(lower) > 2:
        candidates.append(lower[:-1])         # cats → cat, makes → make
    # Past tense (-ed)
    if lower.endswith("ied") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")   # tried → try
    elif lower.endswith("ed") and len(lower) > 3:
        candidates.append(lower[:-2])         # walked → walk
        candidates.append(lower[:-1])         # baked → bake (drop only d)
    # Present participle / gerund (-ing)
    if lower.endswith("ing") and len(lower) > 4:
        candidates.append(lower[:-3])         # walking → walk
        candidates.append(lower[:-3] + "e")   # baking → bake
        # doubled consonant: running → run, sitting → sit
        if len(lower) >= 5 and lower[-4] == lower[-5]:
            candidates.append(lower[:-4])
    # Comparative (-er) / superlative (-est)
    if lower.endswith("iest") and len(lower) > 5:
        candidates.append(lower[:-4] + "y")   # happiest → happy
    elif lower.endswith("est") and len(lower) > 4:
        candidates.append(lower[:-3])         # biggest → big
        candidates.append(lower[:-2])         # largest → large
    elif lower.endswith("ier") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")   # happier → happy
    elif lower.endswith("er") and len(lower) > 3:
        candidates.append(lower[:-2])         # bigger → big
        candidates.append(lower[:-1])         # larger → large
    return candidates


async def validate_sentence(sentence: str, repos) -> ValidationResult:
    tokens = _tokenize(sentence)
    words_used = []
    non_ket = []
    for i, tok in enumerate(tokens):
        if _is_proper_noun(tok) and i != 0:
            continue
        candidates = _candidate_roots(tok)
        # Skip if any candidate lemma is a function word (e.g. "doing" → "do").
        if any(c in _FUNCTION_WORDS for c in candidates):
            continue
        # First KET-vocab candidate wins (candidates are ordered most-specific
        # first, so an explicit irregular lemma always beats a rule-based one).
        # Use canonical form so stats tracking reconciles with target-word
        # selection (target words come from the CSV with their canonical
        # casing — e.g. the pronoun "I", not "i").
        ket_root = None
        for c in candidates:
            wr = await repos.vocab.get_ket_word_any_context(c)
            if wr is not None:
                ket_root = wr.word   # WordRef.word — canonical form
                break
        if ket_root is not None:
            words_used.append(ket_root)
        else:
            non_ket.append(tok)
    return ValidationResult(
        ok=(len(non_ket) == 0),
        words_used=words_used,
        non_ket_words=non_ket,
    )
