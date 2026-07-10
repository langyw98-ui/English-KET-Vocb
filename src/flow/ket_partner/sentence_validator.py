import json
import re
from os.path import dirname, join
from typing import Optional

from pydantic import BaseModel, Field

from flow.ket_partner.multi_word_target import target_in_sentence

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
    return re.findall(r"[A-Za-z'.-]+", sentence)


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

    `lower` is the punctuation-stripped form so inflection rules like
    `endswith("s")` work on sentence-final tokens (e.g. "runs." → "runs").
    The raw form is still added as a candidate for dictionary entries that
    store punctuation verbatim ("p.m.", "Yeah!").
    """
    raw = token.lower()
    lower = raw.rstrip(".?!")
    candidates = [raw]
    if lower != raw:
        candidates.append(lower)
    # KET dictionary stores some entries with terminal punctuation attached
    # — exclamations ("congratulations!", "Yeah!"), abbreviations ("a.m.",
    # "p.m."). Sentence tokens may carry the punctuation (raw) or not
    # (lower); try both plus the suffix-attached forms.
    for suffix in ("!", "?", "."):
        candidates.append(lower + suffix)
    if lower in _LEMMAS:
        candidates.append(_LEMMAS[lower])
    # Plurals / 3rd-person singular. Try BOTH "-s" and "-es" stems, -s FIRST:
    # a word like "uses" ends in both "s" and "es" — the -s branch yields
    # "use" (correct verb stem) and the -es branch yields "us" (wrong — the
    # pronoun). If -es came first, "us" would match KET and steal the verb's
    # stats. For true -es plurals (watches/boxes), the -s branch produces
    # "watche"/"boxe" which aren't KET, so the lookup falls through to the
    # -es branch's "watch"/"box". First-KET-match wins either way.
    if lower.endswith("ies") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")   # stories → story
    if lower.endswith("s") and len(lower) > 2:
        candidates.append(lower[:-1])         # cats → cat, makes → make, uses → use
    if lower.endswith("es") and len(lower) > 3:
        candidates.append(lower[:-2])         # watches → watch, boxes → box
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


async def validate_sentence(
    sentence: str, repos, target: Optional[str] = None
) -> ValidationResult:
    # Multi-word target awareness: if the caller passed a multi-word target
    # (e.g. "alarm clock") and it appears contiguously in the sentence, treat
    # the phrase as one KET entry rather than tokenizing its constituents.
    # Without this, "alarm" alone isn't KET, so the sentence gets rejected
    # before any post-validation patch can rescue it. Constituent tokens
    # ("alarm", "clock") are skipped during the per-token loop; the target
    # itself is appended once at the end.
    target_constituents: set = set()
    target_present = False
    if target and " " in target.strip() and target_in_sentence(target, sentence):
        target_present = True
        target_constituents = {c.lower().rstrip(".?!") for c in target.split()}

    tokens = _tokenize(sentence)
    words_used = []
    non_ket = []
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().rstrip(".?!")
        if tok_clean in target_constituents:
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
        elif _is_proper_noun(tok) and i != 0:
            # KET lookup failed. If the token starts uppercase and is mid-
            # sentence, treat it as an unknown proper noun (LLM-generated
            # person/place name) and tolerate it. Querying KET first ensures
            # canonical-form KET entries written uppercase (DVD, T-shirt, I,
            # Monday, China) are recognized even at i>0 — previously the
            # _is_proper_noun guard ran first and skip-removed them.
            continue
        else:
            non_ket.append(tok_clean)
    if target_present and target not in words_used:
        words_used.append(target)
    return ValidationResult(
        ok=(len(non_ket) == 0),
        words_used=words_used,
        non_ket_words=non_ket,
    )
