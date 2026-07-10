import re

_PLACEHOLDER_TOKENS = frozenset({
    "somebody", "someone", "something",
    "anybody", "anyone", "anything",
    "nobody", "nothing",
})

_PLACEHOLDER_RE = re.compile(
    r"\b(somebody|someone|something|anybody|anyone|anything|nobody|nothing)\b",
    re.IGNORECASE,
)

# A placeholder substitutes for 1-3 words in the sentence (e.g. "him",
# "my mom", "the tall man"). KET-level sentences rarely need more.
_SUB_PATTERN = r"[A-Za-z']+(?:\s+[A-Za-z']+){0,2}"


def has_placeholder(target: str) -> bool:
    return bool(target) and bool(_PLACEHOLDER_RE.search(target))


def _verb_alternatives(word: str) -> str:
    """Regex alternation matching common inflections of an English verb.

    Used for the first literal part of a placeholder phrase (typically the
    verb, e.g. 'give' in 'give somebody a call') so that 'gives' / 'giving'
    / 'given' still match. Irregular past tense ('gave') is not derivable
    from rules and is not covered — accepted as a known limitation since
    KET-level prompts mostly elicit the base or -s form.
    """
    if word.endswith("e") and len(word) > 2:
        return f"(?:{re.escape(word)}|{re.escape(word)}s|{re.escape(word[:-1])}ing|{re.escape(word[:-1])}en)"
    return f"(?:{re.escape(word)}|{re.escape(word)}s|{re.escape(word)}ed|{re.escape(word)}ing)"


def build_target_pattern(target: str) -> re.Pattern:
    if not has_placeholder(target):
        return re.compile(re.escape(target), re.IGNORECASE)
    tokens = target.lower().split()
    parts = []
    first_literal_seen = False
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(r"\s+")
        if tok in _PLACEHOLDER_TOKENS:
            parts.append(_SUB_PATTERN)
        elif not first_literal_seen:
            parts.append(_verb_alternatives(tok))
            first_literal_seen = True
        else:
            parts.append(re.escape(tok))
    return re.compile("".join(parts), re.IGNORECASE)


def target_in_sentence(target: str, sentence: str) -> bool:
    return bool(build_target_pattern(target).search(sentence))


def find_placeholder(target: str) -> str:
    m = _PLACEHOLDER_RE.search(target or "")
    return m.group(0) if m else ""
