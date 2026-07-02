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
