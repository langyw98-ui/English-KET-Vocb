"""Sentence generation + validation orchestration.

Extracted from KETPartnerAgent to enable isolated unit testing.
Stateless pure functions; takes all dependencies as parameters.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.multi_word_target import target_in_sentence
from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.sentence_generator import generate_sentence
from flow.ket_partner.sentence_naturalness import check_naturalness
from flow.ket_partner.sentence_validator import ValidationResult, validate_sentence
from flow.ket_partner.vocab_selector import select_target_word


async def validate_and_categorize(
    llm_smart: BaseChatModel,
    sentence: str,
    target: str,
    age: int,
    repos: KETPartnerRepos,
    avoid_sentences: list[str],
) -> dict:
    """Copy body from agent.py:213-268 (KETPartnerAgent._validate_and_categorize).
    Replace self.llm_smart -> llm_smart. Returns dict with keys:
    result, passed, reason_kind, reason_detail, non_ket_words, non_ket_count,
    is_duplicate, is_target_split, sentence.
    """
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
    """Copy body from agent.py:270-380 (KETPartnerAgent._generate_with_fallback).
    Replace self.llm_smart / self.config -> llm_smart / config.
    """
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
) -> None:
    """In-place patch result.words_used / non_ket_words for multi-word targets.
    Extracted inline block from agent.py:174-193.
    """
    if (
        target
        and target not in result.words_used
        and target_in_sentence(target, sentence)
    ):
        result.words_used.append(target)
        constituents = {c.lower() for c in target.split()}
        result.words_used = [
            w for w in result.words_used
            if w == target or w.lower() not in constituents
        ]
        result.non_ket_words = [
            w for w in result.non_ket_words
            if w.lower() not in constituents
        ]
        logger.debug(
            f"multi-word target patch: added '{target}', "
            f"final words_used={result.words_used}, "
            f"non_ket_words={result.non_ket_words}"
        )
