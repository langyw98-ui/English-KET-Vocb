import asyncio
import json
import random
import re
from dataclasses import dataclass
from os.path import dirname, join

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.vocab_domain import select_target_word

# sentence_domain 内所有 LLM 调用的可重试外部失败类型。
# 严格按 CLAUDE.md §1.5:只含具体外部失败,不含 ValueError/AttributeError/TypeError
# 等代码 bug 类型——那些必须直接暴露被测试捕获。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,          # openai SDK 的所有 API 异常基类(APITimeoutError/APIConnectionError/RateLimitError 等)
    asyncio.TimeoutError,     # asyncio.wait_for 超时
    ValidationError,          # pydantic Schema 校验失败(LLM 返回畸形结构)
)

# ---------------------------------------------------------------------------
# Multi-word Target Matching
# ---------------------------------------------------------------------------

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
    """Regex alternation matching common inflections of an English verb."""
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


# ---------------------------------------------------------------------------
# Sentence Generator
# ---------------------------------------------------------------------------

_GENERATOR_SYSTEM = """You write ONE English sentence for a {age}-year-old Chinese kid to translate.

Constraints:
- {min_words}-{max_words} words, single sentence.
- Naturally include the target: "{target}".{multi_word_note}
- Vary scaffolding words. Don't reuse words from recent sentences: {recent}.
- Do NOT output any of these exact sentences (must differ in wording, subject, or scene): {avoid}
- Prefer words the kid has likely mastered.
- All words must be from KET vocabulary (will be validated).
- Must be NATURAL and make real-world sense. Subject-verb-object must reflect how things actually behave. No nonsense like "ice cream makes my nose move" or "the book sings".
- Playful or imaginative situations are fine, but only if internally coherent.
- NO emoji, NO Chinese.
{history_block}{non_ket_block}{target_split_block}{context_block}
Output: just the English sentence, nothing else.
"""

_HISTORY_BLOCK = """
Your previous attempts were all rejected. Do NOT repeat the same mistakes — address each rejection's underlying issue (do not just change the subject or surrounding words while keeping the same problematic collocation / word choice):
{history}
"""

_NON_KET_BLOCK = """
Previous attempt(s) used these non-KET words — do NOT use them again, pick KET alternatives instead:
{words}
"""

_TARGET_SPLIT_BLOCK = """
At least one previous attempt SPLIT the target phrase — its words did NOT appear contiguously in the sentence.
The target "{target}" MUST appear as a single inseparable unit, with its words side-by-side in the same order.
Rewrite the sentence so the target phrase is intact.
"""

_CONTEXT_BLOCK = """
Use "{target}" specifically in this sense: {context}.
"""

_MULTI_WORD_NOTE = " The target is a MULTI-WORD phrase — its words MUST appear contiguously in the sentence, side-by-side in the same order. Do NOT split them with other words."

_PLACEHOLDER_MULTI_WORD_NOTE = ' The target is a multi-word phrase with a PLACEHOLDER — "{placeholder}" MUST be REPLACED by a concrete noun, name, or pronoun (e.g. him / my mom / the teacher). The OTHER words must appear contiguously and in the same order, with the replacement occupying exactly the placeholder\'s slot. Do NOT keep "{placeholder}" in the sentence.'


def _select_multi_word_note(target: str) -> str:
    if not target or " " not in target.strip():
        return ""
    if has_placeholder(target):
        return _PLACEHOLDER_MULTI_WORD_NOTE.format(placeholder=find_placeholder(target))
    return _MULTI_WORD_NOTE


def _format_history(attempts: list) -> str:
    if not attempts:
        return ""
    lines = []
    for i, a in enumerate(attempts, 1):
        lines.append(f'  {i}. "{a["sentence"]}" — Reason: {a["reason_detail"]}')
    return "\n".join(lines)


async def generate_sentence(
    llm,
    target: str,
    recent_scaffolding: list,
    age: int,
    min_words: int,
    max_words: int,
    avoid_sentences: list | None = None,
    prior_attempts: list | None = None,
    avoid_non_ket_words: list | None = None,
    target_context: str = "",
) -> str:
    creative = llm.bind(temperature=0.8)
    avoid_sentences = avoid_sentences or []
    prior_attempts = prior_attempts or []
    avoid_non_ket_words = avoid_non_ket_words or []
    history_block = _HISTORY_BLOCK.format(history=_format_history(prior_attempts)) if prior_attempts else ""
    non_ket_block = _NON_KET_BLOCK.format(words=", ".join(avoid_non_ket_words)) if avoid_non_ket_words else ""
    multi_word_note = _select_multi_word_note(target)
    target_split_block = (
        _TARGET_SPLIT_BLOCK.format(target=target)
        if any(a.get("reason_kind") == "target_split" for a in prior_attempts)
        else ""
    )
    context_block = (
        _CONTEXT_BLOCK.format(target=target, context=target_context)
        if target_context else ""
    )
    system_text = _GENERATOR_SYSTEM.format(
        age=age,
        min_words=min_words,
        max_words=max_words,
        target=target,
        recent=", ".join(recent_scaffolding) or "(none yet)",
        avoid="\n".join(f"  - {s}" for s in avoid_sentences) or "(none yet)",
        multi_word_note=multi_word_note,
        history_block=history_block,
        non_ket_block=non_ket_block,
        target_split_block=target_split_block,
        context_block=context_block,
    )
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"Write a sentence using target word '{target}'."),
    ]
    logger.debug(f"generate_sentence: {messages}")
    try:
        response = await creative.ainvoke(messages)
        logger.debug(f"generate_sentence: {response.content.strip()}")
        return response.content.strip()
    except _LLM_RETRYABLE as e:
        logger.warning(
            f"generate_sentence failed (using fallback template): {e}", exc_info=True
        )
        templates = [
            f"I see a {target}.",
            f"The {target} is very nice.",
            f"Look at the {target} over there.",
            f"I like the {target}.",
            f"This is a good {target}.",
        ]
        return random.choice(templates)


# ---------------------------------------------------------------------------
# Sentence Validator
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True, slots=True)
class SentenceGenerationResult:
    """generate_with_fallback 的最终返回类型。

    替代原裸 4 元组 (sentence, result, target, context),让调用方按命名属性访问,
    位置错配在编译期就能暴露。
    """
    sentence: str
    result: ValidationResult
    target: str
    context: str


@dataclass(frozen=True, slots=True)
class _RetryOuter:
    """_switch_target_or_accept 的内部信号:已切换 target,请求外层 while 重试。

    不对外暴露(下划线前缀)。
    """
    target: str
    context: str


def _tokenize(sentence: str) -> list:
    return re.findall(r"[A-Za-z'.-]+", sentence)


def _is_proper_noun(token: str) -> bool:
    return token[:1].isupper()


def _candidate_roots(token: str) -> list:
    raw = token.lower()
    lower = raw.rstrip(".?!")
    candidates = []
    orig = token.rstrip(".?!")
    if orig:
        candidates.append(orig)
    candidates.append(raw)
    if lower != raw:
        candidates.append(lower)
    for suffix in ("!", "?", "."):
        candidates.append(lower + suffix)
    if lower in _LEMMAS:
        candidates.append(_LEMMAS[lower])
    if lower.endswith("ies") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")
    if lower.endswith("s") and len(lower) > 2:
        candidates.append(lower[:-1])
    if lower.endswith("es") and len(lower) > 3:
        candidates.append(lower[:-2])
    if lower.endswith("ied") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")
    elif lower.endswith("ed") and len(lower) > 3:
        candidates.append(lower[:-2])
        candidates.append(lower[:-1])
    if lower.endswith("ing") and len(lower) > 4:
        candidates.append(lower[:-3])
        candidates.append(lower[:-3] + "e")
        if len(lower) >= 5 and lower[-4] == lower[-5]:
            candidates.append(lower[:-4])
    if lower.endswith("iest") and len(lower) > 5:
        candidates.append(lower[:-4] + "y")
    elif lower.endswith("est") and len(lower) > 4:
        candidates.append(lower[:-3])
        candidates.append(lower[:-2])
    elif lower.endswith("ier") and len(lower) > 4:
        candidates.append(lower[:-3] + "y")
    elif lower.endswith("er") and len(lower) > 3:
        candidates.append(lower[:-2])
        candidates.append(lower[:-1])
    return candidates


async def validate_sentence(
    sentence: str, repos: KETPartnerRepos, target: str | None = None
) -> ValidationResult:
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
        if any(c in _FUNCTION_WORDS for c in candidates):
            continue
        ket_root = None
        for c in candidates:
            wr = await repos.vocab.get_ket_word_any_context(c)
            if wr is not None:
                ket_root = wr.word
                break
        if ket_root is not None:
            words_used.append(ket_root)
        elif _is_proper_noun(tok) and i != 0:
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


# ---------------------------------------------------------------------------
# Sentence Naturalness Evaluation
# ---------------------------------------------------------------------------

_NATURALNESS_SYSTEM = """You judge whether ONE English sentence for a {age}-year-old Chinese kid is NATURAL and makes real-world sense.

Accept (ok=true) if the sentence describes a plausible situation, even if playful or imaginative.
Reject (ok=false) if any of the following apply:

1. Subject-verb-object combinations violate how things actually behave AND the sentence is not a coherent fantasy.
   - Reject: "The cold ice cream makes my nose move." — ice cream does not make noses move.
   - Reject: "The book sings a loud song." — books do not sing.

2. Semantic redundancy / tautology — an adjective or modifier that is inherent to the noun it modifies.
   - Reject: "The dog moves the wet water off its hair." — water is inherently wet; "wet water" is not natural English.
   - Reject: "The cold ice is in the cup." — ice is inherently cold.
   - Reject: "The hot fire burns the wood." — fire is inherently hot.
   - Accept: "The dog shakes the water off its fur." — natural.
   - Accept: "The ice in the cup is melting." — natural.

3. Collocation errors — word combinations that are grammatically valid but native speakers would never use.
   - Reject: "The dog moves the water off its hair." — English speakers say "shakes off" or "dries", not "moves off"; dogs have "fur" not "hair".

Accept examples (natural):
- "The cold ice cream makes my teeth hurt."
- "The funny cat rested in my bed."
- "The monkey eats a yellow banana."
- "The little bird sings a happy song."

When rejecting, `reason` must be ONE short sentence identifying which category (1, 2, or 3) applies and why.
When accepting, leave `reason` empty.

Output only the structured fields.
"""


class NaturalnessResult(BaseModel):
    ok: bool
    reason: str = ""


async def check_naturalness(llm, sentence: str, age: int = 8) -> NaturalnessResult:
    structured = llm.with_structured_output(NaturalnessResult, method="function_calling")
    messages = [
        SystemMessage(content=_NATURALNESS_SYSTEM.format(age=age)),
        HumanMessage(content=f"Sentence: {sentence}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except _LLM_RETRYABLE as e:
        logger.warning(
            f"check_naturalness failed: {e}; accepting by default", exc_info=True
        )
        return NaturalnessResult(ok=True, reason="")


# ---------------------------------------------------------------------------
# Sentence Orchestration
# ---------------------------------------------------------------------------

async def validate_and_categorize(
    llm_smart: BaseChatModel,
    sentence: str,
    target: str,
    age: int,
    repos: KETPartnerRepos,
    avoid_sentences: list[str],
) -> dict:
    result = await validate_sentence(sentence, repos, target=target)
    is_duplicate = sentence in avoid_sentences
    non_ket_count = len(result.non_ket_words)
    is_target_split = (
        bool(target)
        and " " in target.strip()
        and not target_in_sentence(target, sentence)
    )

    passed = False
    reason_kind = None
    reason_detail = ""

    if non_ket_count <= 1 and not is_duplicate and not is_target_split:
        if non_ket_count == 0:
            naturalness = await check_naturalness(llm_smart, sentence, age=age)
            logger.debug(
                f"validate_sentence: {result} duplicate={is_duplicate} "
                f"target_split={is_target_split} naturalness_ok={naturalness.ok}"
            )
            if naturalness.ok:
                passed = True
            else:
                reason_kind = "naturalness"
                reason_detail = f"unnatural expression — {naturalness.reason}"
        else:
            logger.debug(f"{result} accept: 1 non-KET word will annotate")
            passed = True
    else:
        logger.debug(
            f"validate_sentence: {result} duplicate={is_duplicate} target_split={is_target_split}"
        )
        if is_target_split:
            reason_kind = "target_split"
            reason_detail = f"split the multi-word target '{target}' — words must be contiguous"
        elif is_duplicate:
            reason_kind = "duplicate"
            reason_detail = "word-for-word duplicate of a recent sentence"
        else:
            reason_kind = "non_ket_overflow"
            reason_detail = f"non-KET words {result.non_ket_words} exceed the limit (max 1 allowed)"

    return {
        "result": result,
        "passed": passed,
        "reason_kind": reason_kind,
        "reason_detail": reason_detail,
        "non_ket_words": list(result.non_ket_words),
        "non_ket_count": non_ket_count,
        "is_duplicate": is_duplicate,
        "is_target_split": is_target_split,
        "sentence": sentence,
    }


async def generate_with_fallback(
    llm_smart: BaseChatModel,
    initial_target: str,
    initial_context: str,
    avoid_words: list[str],
    avoid_sentences: list[str],
    age: int,
    profile: dict,
    repos: KETPartnerRepos,
    config: KetConfig,
) -> tuple[str, ValidationResult, str, str]:
    target = initial_target
    context = initial_context
    word_switched = False

    while True:
        attempts: list[dict] = []
        seen_non_ket_words: list = []

        def _regen():
            return generate_sentence(
                llm_smart,
                target=target,
                recent_scaffolding=avoid_words,
                age=age,
                min_words=config.sentence.min_words,
                max_words=config.sentence.max_words,
                avoid_sentences=avoid_sentences,
                prior_attempts=attempts,
                avoid_non_ket_words=seen_non_ket_words,
                target_context=context,
            )

        sentence = await _regen()
        result = None

        for _ in range(config.validate_retry_limit):
            check = await validate_and_categorize(
                llm_smart, sentence, target, age, repos, avoid_sentences
            )
            result = check["result"]
            if check["passed"]:
                return sentence, result, target, context
            attempts.append({
                "sentence": sentence,
                "reason_kind": check["reason_kind"],
                "reason_detail": check["reason_detail"],
                "non_ket_words": check["non_ket_words"],
                "non_ket_count": check["non_ket_count"],
            })
            for w in check["non_ket_words"]:
                if w not in seen_non_ket_words:
                    seen_non_ket_words.append(w)
            sentence = await _regen()

        check = await validate_and_categorize(
            llm_smart, sentence, target, age, repos, avoid_sentences
        )
        result = check["result"]
        if check["passed"]:
            return sentence, result, target, context
        attempts.append({
            "sentence": sentence,
            "reason_kind": check["reason_kind"],
            "reason_detail": check["reason_detail"],
            "non_ket_words": check["non_ket_words"],
            "non_ket_count": check["non_ket_count"],
        })

        overflow_attempts = [a for a in attempts if a["reason_kind"] == "non_ket_overflow"]
        all_naturalness = bool(attempts) and all(
            a["reason_kind"] == "naturalness" for a in attempts
        )

        if overflow_attempts:
            best = min(reversed(overflow_attempts), key=lambda a: a["non_ket_count"])
            sentence = best["sentence"]
            result = await validate_sentence(sentence, repos, target=target)
            logger.warning(
                f"sentence validation: accepting non-KET overflow draft after "
                f"{len(attempts)} attempts (non_ket_count={len(result.non_ket_words)}); "
                f"sentence={sentence!r}"
            )
            return sentence, result, target, context
        elif all_naturalness and not word_switched:
            logger.info(
                f"all {len(attempts)} attempts failed on naturalness; "
                f"switching target word from '{target}'"
            )
            new_ref = await select_target_word(repos, profile, config)
            if new_ref is None or new_ref.word == target:
                logger.warning(
                    f"could not find a different target word; "
                    f"accepting final draft: {sentence!r}"
                )
                return sentence, result, target, context
            target = new_ref.word
            context = new_ref.context
            word_switched = True
            continue
        else:
            reasons = []
            if check["non_ket_count"] > 1:
                reasons.append(f"{check['non_ket_count']} non-KET word(s): {check['non_ket_words']}")
            if check["is_duplicate"]:
                reasons.append("duplicate of a recent sentence")
            if check["is_target_split"]:
                reasons.append(f"multi-word target '{target}' split apart")
            if not reasons:
                reasons.append(f"naturalness: {check['reason_detail']}")
            logger.warning(
                f"sentence validation failed after {len(attempts)} attempts; "
                f"accepting current draft — reasons: {('; '.join(reasons)) or 'unknown'}; "
                f"sentence={sentence!r}"
            )
            return sentence, result, target, context


def apply_multiword_target_patch(
    target: str,
    sentence: str,
    result: ValidationResult,
) -> ValidationResult:
    """对多词 target 应用补丁:确保 words_used 和 non_ket_words 正确反映多词边界。

    纯函数:不改入参 result,返回新的 ValidationResult。
    调用方必须使用返回值,如:
        result = apply_multiword_target_patch(target, sentence, result)
    """
    if (
        not target
        or target in result.words_used
        or not target_in_sentence(target, sentence)
    ):
        return result

    # 计算新的 words_used 和 non_ket_words(原逻辑保留,但改为构造新列表)
    new_words_used = [*result.words_used, target]
    constituents = {c.lower() for c in target.split()}
    new_words_used = [
        w for w in new_words_used
        if w == target or w.lower() not in constituents
    ]
    new_non_ket_words = [
        w for w in result.non_ket_words
        if w.lower() not in constituents
    ]
    logger.debug(
        f"multi-word target patch: added '{target}', "
        f"final words_used={new_words_used}, "
        f"non_ket_words={new_non_ket_words}"
    )
    return result.model_copy(update={
        "words_used": new_words_used,
        "non_ket_words": new_non_ket_words,
    })
