import random
from typing import Optional

from flow.ket_partner.config import KetConfig
from flow.ket_partner.db import Repos


def _compute_refill_mode(learning_count: int, current_flag: int, low: int, high: int) -> int:
    if learning_count >= high:
        return 0
    if learning_count < low:
        return 1
    return current_flag


async def rotate_topic(repos: Repos, current: Optional[str]) -> Optional[str]:
    candidates = await repos.vocab.topics_with_unmastered(exclude=current)
    return candidates[0] if candidates else current


async def select_target_word(
    repos: Repos, profile: dict, config: KetConfig
) -> Optional[str]:
    low = config.vocab_refill.low_watermark
    high = config.vocab_refill.high_watermark
    interval = config.vocab_refill.interval_turns

    learning_count = await repos.stats.learning_count()
    in_refill = _compute_refill_mode(learning_count, profile["in_refill_mode"], low, high)

    turn = profile["total_turns"]
    # Cold start = pool below low watermark, not "completely empty". The
    # narrower `learning_count == 0` definition closed the gate after turn 1
    # and forced turn 2 to repeat the just-introduced word (only learning word
    # in the pool). Filling the pool to the watermark before letting interval
    # gating kick in guarantees distinct target words across the first turns.
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


async def _pick_new_word(repos: Repos, profile: dict) -> Optional[str]:
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
