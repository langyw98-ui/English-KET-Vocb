from typing import Dict, List

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger

_SYSTEM = """You look up the Chinese meaning of an English word as used in a specific sentence.
Give a concise, kid-friendly Chinese meaning (1-5 characters preferred). Output only the meaning.
"""


class WordMeaning(BaseModel):
    meaning: str


async def lookup_word_meaning(llm, sentence: str, word: str) -> WordMeaning:
    structured = llm.with_structured_output(WordMeaning, method="function_calling")
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Sentence: {sentence}"),
        HumanMessage(content=f"Word: {word}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:
        logger.warning(f"lookup_word_meaning failed: {e}")
        return WordMeaning(meaning=f"({word} 词义查询失败)")


_MULTI_SYSTEM = """You look up the Chinese meaning of several English words as each is used in ONE specific sentence.
For each word, give a concise, kid-friendly Chinese meaning (1-5 characters preferred) that fits how the word is used in that sentence.

Output one entry per requested word. The set of words in your output MUST exactly match the set of words given.
"""


class _SingleMeaning(BaseModel):
    word: str
    meaning: str


class WordMeanings(BaseModel):
    meanings: List[_SingleMeaning] = Field(default_factory=list)


async def lookup_word_meanings(llm, sentence: str, words: List[str]) -> List[Dict[str, str]]:
    """Batch lookup of Chinese meanings for several non-KET words in the
    context of one sentence. Used by generate_sentence_node to annotate
    accepted sentences that contain up to 1 (or, after retries, more)
    non-KET words. Returns a list of {word, meaning} dicts in the same
    order as `words`.
    """
    if not words:
        return []
    structured = llm.with_structured_output(WordMeanings, method="function_calling")
    messages = [
        SystemMessage(content=_MULTI_SYSTEM),
        HumanMessage(content=f"Sentence: {sentence}"),
        HumanMessage(content=f"Words (look up each, in this sentence's context): {', '.join(words)}"),
    ]
    try:
        result = await structured.ainvoke(messages)
        by_word = {m.word: m.meaning for m in result.meanings}
        # Preserve input order; fall back to "" if LLM dropped a word.
        return [{"word": w, "meaning": by_word.get(w, "")} for w in words]
    except Exception as e:
        logger.warning(f"lookup_word_meanings failed: {e}")
        return [{"word": w, "meaning": ""} for w in words]


_SENTENCE_SYSTEM = """Translate the English sentence into natural, kid-friendly Chinese.
Output only the translation, nothing else."""


class SentenceTranslation(BaseModel):
    translation: str


async def lookup_sentence_translation(llm, sentence: str) -> SentenceTranslation:
    """Used on the idk path — the kid doesn't know how to translate, so
    we show the full correct Chinese translation rather than just the
    target word's meaning.
    """
    structured = llm.with_structured_output(SentenceTranslation, method="function_calling")
    messages = [
        SystemMessage(content=_SENTENCE_SYSTEM),
        HumanMessage(content=f"Sentence: {sentence}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:
        logger.warning(f"lookup_sentence_translation failed: {e}")
        return SentenceTranslation(translation="(翻译失败)")
