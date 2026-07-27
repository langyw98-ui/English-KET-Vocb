import re

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger

_SYSTEM = """你查询一个英语单词在特定句子中的中文意思。
请给出简洁、适合儿童的中文释义（建议 1-5 个字）。

关键要求：输出必须是中文汉字。禁止把英语单词原样返回。
- "sea" → 海（正确）
- "sea" → sea（禁止，这是英语单词本身，不是它的意思）
- "apple" → 苹果（正确）
- "apple" → apple（禁止）

只输出中文释义，不要输出任何其他内容。
"""

# Matches any Unicode CJK Unified Ideograph. Used to verify the LLM actually
# returned Chinese rather than echoing the English word back. Observed in
# production: kid asked about "sea", the LLM returned meaning="sea" — the
# kid-facing message rendered as `「sea」的意思是「sea」`, completely useless.
# Chinese prompts make this rarer but not impossible, so the check stays as
# a defense in depth.
_CJK_RE = re.compile(r"[一-鿿]")


def _has_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


class WordMeaning(BaseModel):
    meaning: str


async def lookup_word_meaning(
    llm, sentence: str, word: str, context: str = ""
) -> WordMeaning:
    structured = llm.with_structured_output(WordMeaning, method="function_calling")
    word_line = f"单词：{word}"
    if context:
        word_line += f"（in the sense of {context}）"
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"句子：{sentence}"),
        HumanMessage(content=word_line),
    ]
    try:
        result = await structured.ainvoke(messages)
        # Safety net: if the LLM echoed the English word (no Chinese chars),
        # retry once — non-determinism often produces a correct translation
        # on the second call. If the retry also lacks Chinese, fall back to
        # a kid-visible failure marker rather than showing the English echo.
        if not _has_chinese(result.meaning):
            logger.debug(
                f"lookup_word_meaning: first response lacks Chinese ({result.meaning!r}); retrying"
            )
            result = await structured.ainvoke(messages)
            if not _has_chinese(result.meaning):
                logger.warning(
                    f"lookup_word_meaning: '{word}' lookup returned no Chinese after retry "
                    f"(last={result.meaning!r}); using fallback"
                )
                return WordMeaning(meaning=f"({word} 词义查询失败)")
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning(f"lookup_word_meaning failed: {e}")
        return WordMeaning(meaning=f"({word} 词义查询失败)")


_MULTI_SYSTEM = """你查询几个英语单词在同一个句子中的中文意思。
每个单词请给出简洁、适合儿童的中文释义（建议 1-5 个字），需符合该单词在句子中的实际用法。

每个被查询的单词输出一条记录。输出的单词集合必须与输入完全一致，禁止增删或遗漏单词。
"""


class _SingleMeaning(BaseModel):
    word: str
    meaning: str


class WordMeanings(BaseModel):
    meanings: list[_SingleMeaning] = Field(default_factory=list)


async def lookup_word_meanings(llm, sentence: str, words: list[str]) -> list[dict[str, str]]:
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
        HumanMessage(content=f"句子：{sentence}"),
        HumanMessage(content=f"单词（在句子语境中查询每一个）：{', '.join(words)}"),
    ]
    try:
        result = await structured.ainvoke(messages)
        by_word = {m.word: m.meaning for m in result.meanings}
        # Preserve input order; fall back to "" if LLM dropped a word.
        return [{"word": w, "meaning": by_word.get(w, "")} for w in words]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"lookup_word_meanings failed: {e}")
        return [{"word": w, "meaning": ""} for w in words]


_SENTENCE_SYSTEM = """把这个英语句子翻译成自然、适合儿童的中文。
只输出翻译结果，不要输出任何其他内容。"""


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
        HumanMessage(content=f"句子：{sentence}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"lookup_sentence_translation failed: {e}")
        return SentenceTranslation(translation="(翻译失败)")
