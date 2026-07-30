import tempfile

import pytest

from flow.ket_partner.vocab_domain import apply_mastery_updates
from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos


@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic,context\n"
        "cat,n,Animals,\n"
        "dog,n,Animals,\n"
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
    """Spec §9.1: when the target word has a non-empty context, the +1
    delta for the correct target must land at (target, context), not at
    (target, '')."""
    # Add a (smart, clever) vocab row so the apply_delta orphan guard
    # accepts context='clever'.
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
    # Default-sense row must NOT exist (orphan guard skipped it).
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
