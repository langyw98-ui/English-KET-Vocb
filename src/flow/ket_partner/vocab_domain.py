from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.state import (
    ASKS_MEANING,
    IDK,
    TRANSLATION,
    BTPKetState,
)

if TYPE_CHECKING:
    from src.persistence.models import WordRef


# vocab_domain 内所有 LLM 调用的可重试外部失败类型。
# 严格按 CLAUDE.md §1.5:只含具体外部失败,不含 ValueError/AttributeError/TypeError
# 等代码 bug 类型——那些必须直接暴露被测试捕获。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,          # openai SDK 的所有 API 异常基类(APITimeoutError/APIConnectionError/RateLimitError 等)
    asyncio.TimeoutError,     # asyncio.wait_for 超时
    ValidationError,          # pydantic Schema 校验失败(LLM 返回畸形结构)
)


# ---------------------------------------------------------------------------
# Vocabulary Selection & Rotation
# ---------------------------------------------------------------------------

def _compute_refill_mode(learning_count: int, current_flag: int, low: int, high: int) -> int:
    if learning_count >= high:
        return 0
    if learning_count < low:
        return 1
    return current_flag


async def rotate_topic(repos: KETPartnerRepos, current: str | None) -> str | None:
    candidates = await repos.vocab.topics_with_unmastered(exclude=current)
    return candidates[0] if candidates else current


async def select_target_word(
    repos: KETPartnerRepos, profile: dict, config: KetConfig
) -> WordRef | None:
    low = config.vocab_refill.low_watermark
    high = config.vocab_refill.high_watermark
    interval = config.vocab_refill.interval_turns

    learning_count = await repos.stats.learning_count()
    in_refill = _compute_refill_mode(learning_count, profile["in_refill_mode"], low, high)

    turn = profile["total_turns"]
    cold_start = learning_count < low
    if in_refill and (cold_start or (turn - profile["last_new_word_turn"]) >= interval):
        target = await _pick_new_word(repos, profile)
        if target is not None:
            await repos.profile.update(
                last_new_word_turn=turn,
                in_refill_mode=in_refill,
                current_topic=profile["current_topic"],
            )
            return target

    practice = await repos.stats.oldest_learning_word()
    await repos.profile.update(in_refill_mode=in_refill)
    return practice


async def _pick_new_word(repos: KETPartnerRepos, profile: dict) -> WordRef | None:
    topic = profile["current_topic"]
    if topic:
        candidates = await repos.vocab.words_in_topic_without_stats(topic)
        if candidates:
            return candidates[0]

    new_topic = await rotate_topic(repos, topic)
    if new_topic and new_topic != topic:
        await repos.profile.update(current_topic=new_topic)
        profile["current_topic"] = new_topic
        candidates = await repos.vocab.words_in_topic_without_stats(new_topic)
        if candidates:
            return candidates[0]

    mopup = await repos.vocab.unexposed_notopic_words()
    if mopup:
        return mopup[0]

    return await repos.stats.oldest_learning_word()


# ---------------------------------------------------------------------------
# Word Meaning & Sentence Translation Lookups
# ---------------------------------------------------------------------------

_LOOKUP_SYSTEM = """你查询一个英语单词在特定句子中的中文意思。
请给出简洁、适合儿童的中文释义（建议 1-5 个字）。

关键要求：输出必须是中文汉字。禁止把英语单词原样返回。
- "sea" → 海（正确）
- "sea" → sea（禁止，这是英语单词本身，不是它的意思）
- "apple" → 苹果（正确）
- "apple" → apple（禁止）

只输出中文释义，不要输出任何其他内容。
"""

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

    content_parts = []
    if sentence and word.lower() in sentence.lower():
        content_parts.append(f"句子：{sentence}")
    content_parts.append(word_line)

    messages = [
        SystemMessage(content=_LOOKUP_SYSTEM),
        HumanMessage(content="\n".join(content_parts)),
    ]
    try:
        result = await structured.ainvoke(messages)
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
    except _LLM_RETRYABLE as e:
        logger.warning(f"lookup_word_meaning failed: {e}", exc_info=True)
        return WordMeaning(meaning=f"({word} 词义查询失败)")


_MULTI_LOOKUP_SYSTEM = """你查询几个英语单词在同一个句子中的中文意思。
每个单词请给出简洁、适合儿童的中文释义（建议 1-5 个字），需符合该单词在句子中的实际用法。

每个被查询的单词输出一条记录。输出的单词集合必须与输入完全一致，禁止增删或遗漏单词。
"""


class _SingleMeaning(BaseModel):
    word: str
    meaning: str


class WordMeanings(BaseModel):
    meanings: list[_SingleMeaning] = Field(default_factory=list)


async def lookup_word_meanings(llm, sentence: str, words: list[str]) -> list[dict[str, str]]:
    if not words:
        return []
    structured = llm.with_structured_output(WordMeanings, method="function_calling")
    messages = [
        SystemMessage(content=_MULTI_LOOKUP_SYSTEM),
        HumanMessage(content=f"句子：{sentence}"),
        HumanMessage(content=f"单词（在句子语境中查询每一个）：{', '.join(words)}"),
    ]
    try:
        result = await structured.ainvoke(messages)
        by_word = {m.word: m.meaning for m in result.meanings}
        return [{"word": w, "meaning": by_word.get(w, "")} for w in words]
    except _LLM_RETRYABLE as e:
        logger.warning(f"lookup_word_meanings failed: {e}", exc_info=True)
        return [{"word": w, "meaning": ""} for w in words]


_TRANSLATION_SYSTEM = """把这个英语句子翻译成自然、适合儿童的中文。
只输出翻译结果，不要输出任何其他内容。"""


class SentenceTranslation(BaseModel):
    translation: str


async def lookup_sentence_translation(llm, sentence: str) -> SentenceTranslation:
    structured = llm.with_structured_output(SentenceTranslation, method="function_calling")
    messages = [
        SystemMessage(content=_TRANSLATION_SYSTEM),
        HumanMessage(content=f"句子：{sentence}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except _LLM_RETRYABLE as e:
        logger.warning(f"lookup_sentence_translation failed: {e}", exc_info=True)
        return SentenceTranslation(translation="(翻译失败)")


# ---------------------------------------------------------------------------
# Mastery Updates
# ---------------------------------------------------------------------------

async def apply_mastery_updates(state: BTPKetState, repos: KETPartnerRepos) -> None:
    intent = state.get("intent")
    if intent is None:
        logger.debug("apply_mastery_updates: intent is None, skip")
        return

    if intent == TRANSLATION:
        last_words = state.get("last_sentence_words") or []
        target = state.get("last_target_word")
        target_ctx = state.get("last_target_context") or ""
        wrong = {w["word"] for w in state.get("wrong_words") or []}
        overall_correct = state.get("overall_correct")
        neutral_all = (not wrong) and (overall_correct is False)
        if neutral_all:
            return
        for w in last_words:
            ctx = target_ctx if w == target else ""
            delta = -1 if w in wrong else 1
            await repos.stats.apply_delta(w, context=ctx, delta=delta, exposed=False)
    elif intent == IDK:
        target = state.get("last_target_word")
        if target:
            target_ctx = state.get("last_target_context") or ""
            await repos.stats.apply_delta(target, context=target_ctx, delta=-1, exposed=False)
    elif intent == ASKS_MEANING:
        asked = state.get("asked_word")
        if asked:
            target = state.get("last_target_word")
            target_ctx = state.get("last_target_context") or ""
            wr = await repos.vocab.get_ket_word_any_context(asked)
            if wr:
                ctx = target_ctx if wr.word == target else ""
                await repos.stats.apply_delta(wr.word, context=ctx, delta=-1, exposed=False)
