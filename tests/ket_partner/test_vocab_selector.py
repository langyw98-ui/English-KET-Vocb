import pytest

from flow.ket_partner.config import load_config
from flow.ket_partner.db import init_db
from flow.ket_partner.vocab_selector import select_target_word, rotate_topic


@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic\n"
        "cat,n,Animals\n"
        "dog,n,Animals\n"
        "apple,n,Food\n"
        "the,det,\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    r = await init_db(temp_db_path, csv_path=csv_path)
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_select_new_word_when_refill_and_interval_met(repos):
    cfg = load_config()
    profile = await repos.profile.get()
    profile["in_refill_mode"] = 1
    profile["current_topic"] = "Animals"
    profile["last_new_word_turn"] = 0
    profile["total_turns"] = 5
    await repos.profile.update(
        in_refill_mode=1, current_topic="Animals", last_new_word_turn=0, total_turns=5
    )
    word = await select_target_word(repos, profile, cfg)
    assert word in ("cat", "dog")


@pytest.mark.asyncio
async def test_select_practice_word_when_not_refill(repos):
    cfg = load_config()
    # Pool must be at/above low_watermark so _compute_refill_mode respects
    # the profile's in_refill_mode=0. The fixture has only 4 words, so lower
    # the watermark to 2 (3 learning words sit in [low, high) → flag preserved).
    cfg.vocab_refill.low_watermark = 2
    # is_target=True so each word is a real target word (status='learning'),
    # not passively-exposed scaffolding (which would be 'exposed' and thus
    # invisible to learning_count / oldest_learning_word under the new rule).
    await repos.stats.apply_delta("cat", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("dog", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("apple", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("cat", delta=1, is_target=True)
    profile = await repos.profile.get()
    profile["in_refill_mode"] = 0
    word = await select_target_word(repos, profile, cfg)
    assert word in ("dog", "apple")


@pytest.mark.asyncio
async def test_cold_start_picks_distinct_word_on_second_turn(repos):
    # Regression: cold_start was `learning_count == 0`, so after turn 1 the
    # gate closed and turn 2 fell through to oldest_learning_word — which had
    # only the just-introduced word in the pool, forcing the same target twice
    # in a row. With the fix, cold_start = pool below low watermark, so turn 2
    # keeps introducing new words.
    cfg = load_config()
    await repos.profile.update(
        in_refill_mode=1, current_topic="Animals", last_new_word_turn=0, total_turns=0
    )

    # Turn 0: cold pool, picks a new word from Animals topic.
    profile = await repos.profile.get()
    profile["total_turns"] = 0
    word0 = await select_target_word(repos, profile, cfg)
    assert word0 in ("cat", "dog")
    # Simulate persist: target exposure marks it 'learning', turn ticks over.
    await repos.stats.apply_delta(word0, delta=0, exposed=True, is_target=True)

    # Turn 1: pool has 1 word, still below low_watermark. Pre-fix this returned
    # word0 again via oldest_learning_word. Post-fix it must pick a new word.
    profile = await repos.profile.get()
    profile["total_turns"] = 1
    word1 = await select_target_word(repos, profile, cfg)
    assert word1 is not None
    assert word1 != word0, f"turn 2 repeated target {word0!r}; pool should fill first"


@pytest.mark.asyncio
async def test_pool_at_low_watermark_starts_practicing(repos):
    # Boundary: once learning_count reaches low_watermark, cold_start closes
    # and the algorithm routes to oldest_learning_word.
    cfg = load_config()
    low = cfg.vocab_refill.low_watermark
    # Fill the pool to exactly low with distinct target words. The 4-word
    # fixture isn't enough, so override the watermark via direct config edit.
    cfg.vocab_refill.low_watermark = 3
    try:
        await repos.stats.apply_delta("cat", delta=0, exposed=True, is_target=True)
        await repos.stats.apply_delta("dog", delta=0, exposed=True, is_target=True)
        await repos.stats.apply_delta("apple", delta=0, exposed=True, is_target=True)
        # learning_count == low → cold_start False; interval check (0-0=0)<5
        # → falls through to oldest_learning_word.
        await repos.profile.update(
            in_refill_mode=1, current_topic="Animals", last_new_word_turn=0, total_turns=0
        )
        profile = await repos.profile.get()
        word = await select_target_word(repos, profile, cfg)
        # oldest_learning_word by last_seen_at ASC — all three share the same
        # timestamp resolution, so any of them is acceptable; what matters is
        # it's a known learning word, not a fresh pick of "the" (mopup).
        assert word in ("cat", "dog", "apple")
    finally:
        cfg.vocab_refill.low_watermark = low


@pytest.mark.asyncio
async def test_rotate_topic_returns_unmastered(repos):
    await repos.stats.apply_delta("cat", delta=3, exposed=True)
    await repos.stats.apply_delta("dog", delta=3, exposed=True)
    new_topic = await rotate_topic(repos, current="Animals")
    assert new_topic == "Food"


@pytest.mark.asyncio
async def test_mopup_when_all_topic_words_used(repos):
    cfg = load_config()
    await repos.stats.apply_delta("cat", delta=3, exposed=True)
    await repos.stats.apply_delta("dog", delta=3, exposed=True)
    await repos.stats.apply_delta("apple", delta=3, exposed=True)
    profile = await repos.profile.get()
    profile["in_refill_mode"] = 1
    profile["current_topic"] = "Animals"
    profile["last_new_word_turn"] = 0
    profile["total_turns"] = 5
    await repos.profile.update(
        in_refill_mode=1, current_topic="Animals", last_new_word_turn=0, total_turns=5
    )
    word = await select_target_word(repos, profile, cfg)
    assert word == "the"
