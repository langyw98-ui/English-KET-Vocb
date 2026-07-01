import tempfile

import pytest

from flow.ket_partner.config import load_config
from flow.ket_partner.db import init_db
from flow.ket_partner.state import BTPKetState
from flow.ket_partner.nodes import apply_mastery_updates


@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic\n"
        "cat,n,Animals\n"
        "dog,n,Animals\n"
        "big,adj,\n"
        "is,v,\n"
        "the,det,\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    r = await init_db(temp_db_path, csv_path=csv_path)
    yield r
    await r.close()


@pytest.mark.asyncio
async def test_translation_correct_adds_to_all_words(repos):
    state = {
        "intent": "translation",
        "wrong_words": [],
        "last_sentence_words": ["cat", "dog", "big"],
        "last_target_word": "cat",
        "asked_word": None,
    }
    await apply_mastery_updates(state, repos)
    cat = await repos.stats.get("cat")
    assert cat["mastery_score"] == 1
    assert cat["correct_count"] == 1


@pytest.mark.asyncio
async def test_translation_wrong_deducts_specific_words(repos):
    state = {
        "intent": "translation",
        "wrong_words": ["dog"],
        "last_sentence_words": ["cat", "dog"],
        "last_target_word": "cat",
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
        "asked_word": "cat",
    }
    await apply_mastery_updates(state, repos)
    cat = await repos.stats.get("cat")
    assert cat["wrong_count"] == 1


from flow.ket_partner.nodes import format_output_text


def test_format_output_translation_no_wrong():
    state = {
        "intent": "translation",
        "wrong_words": [],
        "correct_meanings": {},
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "The dog runs." in text


def test_format_output_translation_with_wrong():
    state = {
        "intent": "translation",
        "wrong_words": ["cat"],
        "correct_meanings": {"cat": "猫"},
        "last_target_word": "cat",
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "cat" in text
    assert "猫" in text
    assert "The dog runs." in text


def test_format_output_idk():
    state = {
        "intent": "idk",
        "last_target_word": "cat",
        "target_word_meaning": "猫",
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert "猫" in text
    assert "The dog runs." in text


def test_format_output_first_turn_no_feedback():
    state = {
        "intent": None,
    }
    text = format_output_text(state, new_sentence="The dog runs.")
    assert text.strip().startswith("🔤") or "The dog runs." in text
