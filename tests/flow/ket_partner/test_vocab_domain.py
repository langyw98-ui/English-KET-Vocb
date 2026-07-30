import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.config import load_config
from flow.ket_partner.vocab_domain import (
    SentenceTranslation,
    WordMeaning,
    WordMeanings,
    apply_mastery_updates,
    lookup_sentence_translation,
    lookup_word_meaning,
    lookup_word_meanings,
    rotate_topic,
    select_target_word,
)
from src.persistence.bootstrap import init_db
from src.persistence.models import WordRef
from src.persistence.repos import Repos

# ===========================================================================
# Vocab Selector & Topic Rotation Tests
# ===========================================================================

@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic,context\n"
        "cat,n,Animals,\n"
        "dog,n,Animals,\n"
        "apple,n,Food,\n"
        "big,adj,,\n"
        "is,v,,\n"
        "the,det,,\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    r = Repos.for_user(db, "default")
    yield r
    await db.close()


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
    assert word in (WordRef("cat"), WordRef("dog"))


@pytest.mark.asyncio
async def test_select_practice_word_when_not_refill(repos):
    cfg = load_config()
    cfg.vocab_refill.low_watermark = 2
    await repos.stats.apply_delta("cat", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("dog", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("apple", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("cat", delta=1, is_target=True)
    profile = await repos.profile.get()
    profile["in_refill_mode"] = 0
    word = await select_target_word(repos, profile, cfg)
    assert word in (WordRef("dog"), WordRef("apple"))


@pytest.mark.asyncio
async def test_cold_start_picks_distinct_word_on_second_turn(repos):
    cfg = load_config()
    await repos.profile.update(
        in_refill_mode=1, current_topic="Animals", last_new_word_turn=0, total_turns=0
    )

    profile = await repos.profile.get()
    profile["total_turns"] = 0
    word0 = await select_target_word(repos, profile, cfg)
    assert word0 in (WordRef("cat"), WordRef("dog"))
    await repos.stats.apply_delta(word0.word, context=word0.context, delta=0, exposed=True, is_target=True)

    profile = await repos.profile.get()
    profile["total_turns"] = 1
    word1 = await select_target_word(repos, profile, cfg)
    assert word1 is not None
    assert word1 != word0


@pytest.mark.asyncio
async def test_pool_at_low_watermark_starts_practicing(repos):
    cfg = load_config()
    low = cfg.vocab_refill.low_watermark
    cfg.vocab_refill.low_watermark = 3
    try:
        await repos.stats.apply_delta("cat", delta=0, exposed=True, is_target=True)
        await repos.stats.apply_delta("dog", delta=0, exposed=True, is_target=True)
        await repos.stats.apply_delta("apple", delta=0, exposed=True, is_target=True)
        await repos.profile.update(
            in_refill_mode=1, current_topic="Animals", last_new_word_turn=0, total_turns=0
        )
        profile = await repos.profile.get()
        word = await select_target_word(repos, profile, cfg)
        assert word in (WordRef("cat"), WordRef("dog"), WordRef("apple"))
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
    assert word in (WordRef("big"), WordRef("is"), WordRef("the"))


# ===========================================================================
# Word Meaning Lookup Tests
# ===========================================================================

def _make_lookup_llm(return_value):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_lookup_returns_meaning():
    llm = _make_lookup_llm(WordMeaning(meaning="猫"))
    result = await lookup_word_meaning(llm, "The cat is big.", "cat")
    assert result.meaning == "猫"
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_fallback_on_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回默认值,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="sdk timeout",
            request=MagicMock(),
            body=None,
        )
    )
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await lookup_word_meaning(llm, "I see a cat.", "cat")
    assert result.meaning == "(cat 词义查询失败)"
    llm.with_structured_output.assert_called_once_with(WordMeaning, method="function_calling")
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_word_meanings_fallback_on_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回空 meaning 列表,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="sdk timeout",
            request=MagicMock(),
            body=None,
        )
    )
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await lookup_word_meanings(llm, "I see a cat and a dog.", ["cat", "dog"])
    assert result == [
        {"word": "cat", "meaning": ""},
        {"word": "dog", "meaning": ""},
    ]
    llm.with_structured_output.assert_called_once_with(WordMeanings, method="function_calling")
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_sentence_translation_fallback_on_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回默认翻译,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="sdk timeout",
            request=MagicMock(),
            body=None,
        )
    )
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await lookup_sentence_translation(llm, "I see a cat.")
    assert result.translation == "(翻译失败)"
    llm.with_structured_output.assert_called_once_with(
        SentenceTranslation, method="function_calling"
    )
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_retries_when_llm_echoes_english_word():
    llm = _make_lookup_llm(WordMeaning(meaning="海"))
    llm.with_structured_output.return_value.ainvoke.side_effect = [
        WordMeaning(meaning="sea"),
        WordMeaning(meaning="海"),
    ]
    result = await lookup_word_meaning(llm, "My dad loves the sea.", "sea")
    assert result.meaning == "海", (
        f"retry after echo must return Chinese; got {result.meaning!r}"
    )
    assert llm.with_structured_output.return_value.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_lookup_falls_back_when_both_calls_echo_english():
    llm = _make_lookup_llm(WordMeaning(meaning="sea"))
    llm.with_structured_output.return_value.ainvoke.side_effect = [
        WordMeaning(meaning="sea"),
        WordMeaning(meaning="Sea"),
    ]
    result = await lookup_word_meaning(llm, "My dad loves the sea.", "sea")
    assert "失败" in result.meaning, (
        f"persistent English echo must fall back to failure marker; got {result.meaning!r}"
    )
    assert result.meaning.strip().lower() != "sea", (
        f"meaning must not be the bare English echo; got {result.meaning!r}"
    )
    assert llm.with_structured_output.return_value.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_lookup_threads_context_into_prompt():
    llm = _make_lookup_llm(WordMeaning(meaning="聪明"))
    await lookup_word_meaning(llm, "The smart kid.", "smart", context="clever")
    bound = llm.with_structured_output.return_value
    messages = bound.ainvoke.call_args.args[0]
    all_content = " ".join(m.content for m in messages)
    assert "clever" in all_content
    bound.ainvoke.assert_awaited_once()


# ===========================================================================
# Mastery Tracking Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_translation_correct_adds_to_all_words(repos):
    state = {
        "intent": "translation",
        "wrong_words": [],
        "last_sentence_words": ["cat", "dog", "big"],
        "last_target_word": "cat",
        "last_target_context": "",
        "asked_word": None,
    }
    await apply_mastery_updates(state, repos)
    cat = await repos.stats.get("cat")
    assert cat["mastery_score"] == 1
    assert cat["correct_count"] == 1


@pytest.mark.asyncio
async def test_translation_target_uses_real_context(repos):
    await repos.vocab.seed_for_test("smart", context="clever", pos="adj")
    state = {
        "intent": "translation",
        "wrong_words": [],
        "last_sentence_words": ["smart"],
        "last_target_word": "smart",
        "last_target_context": "clever",
        "asked_word": None,
    }
    await apply_mastery_updates(state, repos)
    clever = await repos.stats.get("smart", context="clever")
    assert clever is not None
    assert clever["mastery_score"] == 1
    assert await repos.stats.get("smart", context="") is None


@pytest.mark.asyncio
async def test_translation_wrong_deducts_specific_words(repos):
    state = {
        "intent": "translation",
        "wrong_words": [{"word": "dog", "kid_translation": "x", "correct_translation": "狗"}],
        "last_sentence_words": ["cat", "dog"],
        "last_target_word": "cat",
        "last_target_context": "",
        "asked_word": None,
    }
    await apply_mastery_updates(state, repos)
    cat = await repos.stats.get("cat")
    dog = await repos.stats.get("dog")
    assert cat["mastery_score"] == 1
    assert dog["mastery_score"] == 0
    assert dog["wrong_count"] == 1


@pytest.mark.asyncio
async def test_idk_deducts_target_only(repos):
    await repos.stats.apply_delta("cat", delta=1, exposed=True)
    state = {
        "intent": "idk",
        "wrong_words": None,
        "last_sentence_words": ["cat", "dog"],
        "last_target_word": "cat",
        "last_target_context": "",
        "asked_word": None,
    }
    await apply_mastery_updates(state, repos)
    cat = await repos.stats.get("cat")
    dog = await repos.stats.get("dog")
    assert cat["mastery_score"] == 0
    assert dog is None or dog["mastery_score"] == 0


@pytest.mark.asyncio
async def test_asks_meaning_deducts_asked_word(repos):
    state = {
        "intent": "asks_meaning",
        "wrong_words": None,
        "last_sentence_words": ["cat"],
        "last_target_word": "cat",
        "last_target_context": "",
        "asked_word": "cat",
    }
    await apply_mastery_updates(state, repos)
    cat = await repos.stats.get("cat")
    assert cat["wrong_count"] == 1
