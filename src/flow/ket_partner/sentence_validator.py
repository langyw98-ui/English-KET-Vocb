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


def _to_root(token: str) -> str:
    lower = token.lower()
    return _LEMMAS.get(lower, lower)


def _is_proper_noun(token: str) -> bool:
    return token[:1].isupper()


async def validate_sentence(sentence: str, repos) -> ValidationResult:
    tokens = _tokenize(sentence)
    words_used = []
    non_ket = []
    for i, tok in enumerate(tokens):
        if _is_proper_noun(tok) and i != 0:
            continue
        root = _to_root(tok)
        if root in _FUNCTION_WORDS:
            continue
        if await repos.vocab.is_ket_word(root) or await repos.vocab.is_ket_word(tok.lower()):
            words_used.append(root)
        else:
            non_ket.append(tok)
    return ValidationResult(
        ok=(len(non_ket) == 0),
        words_used=words_used,
        non_ket_words=non_ket,
    )
