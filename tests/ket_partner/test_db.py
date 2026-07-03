import sqlite3
import pytest

from flow.ket_partner.db import _derive_status, init_db, VocabRepo, StatsRepo, ProfileRepo, LogRepo


def test_derive_status_mastered_at_score_cap():
    """mastery_score >= MASTERY_CAP graduates to 'mastered' regardless of
    prior status. CAP=2 means a single correct on top of an exposed/learning
    word (score 1→2) crosses the threshold."""
    assert _derive_status("learning", 2) == "mastered"
    assert _derive_status("exposed", 2) == "mastered"
    assert _derive_status("mastered", 5) == "mastered"  # above cap clamps


def test_derive_status_mastered_demotes_at_score_1_or_below():
    """Demotion path: with CAP=2 there is no absorption buffer, so a single
    wrong answer dropping mastery 2→1 immediately demotes 'mastered' to
    'learning' — re-detection of a drifted word is fast."""
    assert _derive_status("mastered", 1) == "learning"
    assert _derive_status("mastered", 0) == "learning"


def test_derive_status_is_target_promotes_to_learning():
    """is_target=True promotes any sub-cap word to 'learning' (active
    practice). Use mastery=1 to stay below cap so we test the promotion
    branch, not the cap branch."""
    assert _derive_status("exposed", 0, is_target=True) == "learning"
    assert _derive_status("exposed", 1, is_target=True) == "learning"
    assert _derive_status(None, 0, is_target=True) == "learning"  # new row INSERT


def test_derive_status_new_row_not_target_is_exposed():
    assert _derive_status(None, 0) == "exposed"


def test_derive_status_preserves_exposed_and_learning_below_cap():
    """Below cap (mastery < 2): exposed stays exposed, learning stays
    learning. At cap (mastery == 2): graduates to 'mastered'."""
    assert _derive_status("exposed", 1) == "exposed"
    assert _derive_status("exposed", 2) == "mastered"
    assert _derive_status("learning", 1) == "learning"
    assert _derive_status("learning", 2) == "mastered"


@pytest.mark.asyncio
async def test_init_db_creates_tables(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    async with repos.vocab._db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ) as cur:
        rows = await cur.fetchall()
    table_names = {r[0] for r in rows}
    assert "ket_vocabulary" in table_names
    assert "ket_word_topics" in table_names
    assert "vocab_stats" in table_names
    assert "conversation_log" in table_names
    assert "kid_profile" in table_names


@pytest.mark.asyncio
async def test_import_csv_splits_multi_topic(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic\n"
        "cat,n,Animals\n"
        '"bank","n","Finance; Geography"' + "\n"
        "the,det,\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name

    repos = await init_db(temp_db_path, csv_path=csv_path)
    topics_for_bank = await repos.vocab.get_topics_for_word("bank")
    assert set(topics_for_bank) == {"Finance", "Geography"}

    topics_for_cat = await repos.vocab.get_topics_for_word("cat")
    assert topics_for_cat == ["Animals"]

    topics_for_the = await repos.vocab.get_topics_for_word("the")
    assert topics_for_the == []


@pytest.mark.asyncio
async def test_profile_repo_init_creates_single_row(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    profile = await repos.profile.get()
    assert profile["total_turns"] == 0
    assert profile["in_refill_mode"] == 0
    assert profile["current_topic"] is None


@pytest.mark.asyncio
async def test_stats_repo_apply_delta_creates_row(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.apply_delta("cat", delta=1, exposed=True)
    stats = await repos.stats.get("cat")
    assert stats["mastery_score"] == 1
    assert stats["exposed_count"] == 1
    # No is_target → not a target word → 'exposed' (passive exposure with a correct answer)
    assert stats["status"] == "exposed"


@pytest.mark.asyncio
async def test_stats_repo_apply_delta_floor_at_zero(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.apply_delta("cat", delta=-1, exposed=False)
    stats = await repos.stats.get("cat")
    assert stats["mastery_score"] == 0
    assert stats["wrong_count"] == 1


@pytest.mark.asyncio
async def test_stats_repo_apply_delta_caps_mastery_at_cap(temp_db_path):
    """Repeated +1 deltas beyond CAP must stick at CAP=2. Without the cap,
    the score accumulates indefinitely and the kid has to burn many wrong
    answers to demote a previously-mastered word back into the learning
    pool. Capping at 2 keeps the demotion path short: a single wrong answer
    (2→1) demotes immediately."""
    repos = await init_db(temp_db_path, csv_path=None)
    for _ in range(6):
        await repos.stats.apply_delta("cat", delta=1, exposed=True)
    stats = await repos.stats.get("cat")
    assert stats["mastery_score"] == 2, (
        f"mastery_score must cap at 2 (got {stats['mastery_score']})"
    )
    assert stats["status"] == "mastered"
    # exposed_count and correct_count still accumulate — the cap is on
    # mastery_score only, not the audit counters.
    assert stats["exposed_count"] == 6
    assert stats["correct_count"] == 6


@pytest.mark.asyncio
async def test_stats_repo_apply_delta_caps_single_large_delta(temp_db_path):
    """A single delta > CAP (e.g., backfill or test setup) must clamp at CAP=2."""
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.apply_delta("cat", delta=5, exposed=True)
    stats = await repos.stats.get("cat")
    assert stats["mastery_score"] == 2


@pytest.mark.asyncio
async def test_status_transitions(temp_db_path):
    """A scaffolding-only word: stays 'exposed' below mastery CAP=2,
    graduates to 'mastered' at 2, demotes to 'learning' immediately on a
    single wrong answer (no absorption buffer with CAP=2)."""
    repos = await init_db(temp_db_path, csv_path=None)
    # delta=1 exposed=True, no is_target → 'exposed' (passive + correct)
    await repos.stats.apply_delta("cat", delta=1, exposed=True)
    assert (await repos.stats.get("cat"))["status"] == "exposed"
    assert (await repos.stats.get("cat"))["mastery_score"] == 1
    # 1→2: crosses cap → mastered
    await repos.stats.apply_delta("cat", delta=1)
    assert (await repos.stats.get("cat"))["mastery_score"] == 2
    assert (await repos.stats.get("cat"))["status"] == "mastered"
    # 2→1: no absorption buffer with CAP=2 — demotes immediately to learning
    await repos.stats.apply_delta("cat", delta=-1)
    assert (await repos.stats.get("cat"))["mastery_score"] == 1
    assert (await repos.stats.get("cat"))["status"] == "learning"


@pytest.mark.asyncio
async def test_increment_exposed_creates_exposed_row_for_scaffolding(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.increment_exposed("cat")
    stats = await repos.stats.get("cat")
    assert stats["exposed_count"] == 1
    assert stats["mastery_score"] == 0
    assert stats["status"] == "exposed"


@pytest.mark.asyncio
async def test_increment_exposed_creates_learning_row_for_target(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.increment_exposed("cat", is_target=True)
    stats = await repos.stats.get("cat")
    assert stats["exposed_count"] == 1
    assert stats["status"] == "learning"


@pytest.mark.asyncio
async def test_increment_exposed_promotes_existing_exposed_to_learning(temp_db_path):
    """When a word previously seen as scaffolding becomes a target, its
    status must promote from 'exposed' to 'learning'."""
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.increment_exposed("cat")  # scaffolding exposure
    assert (await repos.stats.get("cat"))["status"] == "exposed"
    await repos.stats.increment_exposed("cat", is_target=True)  # now target
    assert (await repos.stats.get("cat"))["status"] == "learning"
    assert (await repos.stats.get("cat"))["exposed_count"] == 2


@pytest.mark.asyncio
async def test_increment_exposed_preserves_learning_on_scaffolding_reexposure(temp_db_path):
    """A word that was target once, then appears as scaffolding in a later
    sentence, must stay 'learning' (target history is sticky)."""
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.increment_exposed("cat", is_target=True)
    await repos.stats.increment_exposed("cat")  # scaffolding re-exposure
    assert (await repos.stats.get("cat"))["status"] == "learning"
    assert (await repos.stats.get("cat"))["exposed_count"] == 2


@pytest.mark.asyncio
async def test_last_ai_message_returns_none_when_no_ai_rows(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    assert await repos.log.last_ai_message() is None


@pytest.mark.asyncio
async def test_last_ai_message_returns_latest_when_no_session_start(temp_db_path):
    """Backward compat: databases populated before session_start was added
    must still return the latest AI row when no marker exists."""
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.log.append("ai", "old sentence", words_used=["old"], turn_id=1)
    await repos.log.append("ai", "newer sentence", words_used=["new"], turn_id=2)
    result = await repos.log.last_ai_message()
    assert result is not None
    assert result["content"] == "newer sentence"


@pytest.mark.asyncio
async def test_last_ai_message_ignores_rows_before_session_start(temp_db_path):
    """Regression: a kid who exits mid-sentence must NOT see that sentence
    restored on restart. append_session_start marks the boundary; all AI
    rows before it become invisible to last_ai_message."""
    repos = await init_db(temp_db_path, csv_path=None)
    # Prior session left an unfinished sentence.
    await repos.log.append("ai", "The unfinished sentence.", words_used=["x"], turn_id=1)
    # REPL restart writes the session_start marker.
    await repos.log.append_session_start()
    # No new AI rows yet in this session.
    assert await repos.log.last_ai_message() is None


@pytest.mark.asyncio
async def test_last_ai_message_returns_only_post_marker_row(temp_db_path):
    """After session_start, new AI rows are visible; pre-marker ones stay hidden."""
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.log.append("ai", "stale from prev session", words_used=["x"], turn_id=1)
    await repos.log.append_session_start()
    await repos.log.append("ai", "fresh this session", words_used=["y"], turn_id=2)
    result = await repos.log.last_ai_message()
    assert result is not None
    assert result["content"] == "fresh this session"
    assert result["words_used"] == ["y"]


@pytest.mark.asyncio
async def test_append_session_start_writes_system_row(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.log.append_session_start()
    async with repos.log._db.execute(
        "SELECT role, content FROM conversation_log ORDER BY id DESC LIMIT 1"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == "system"
    assert row[1] == "session_start"


@pytest.mark.asyncio
async def test_oldest_learning_word_returns_none_when_empty(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    assert await repos.stats.oldest_learning_word() is None


@pytest.mark.asyncio
async def test_oldest_learning_word_prefers_learning_over_exposed(temp_db_path):
    """When both pools exist, learning wins — exposed is only the fallback."""
    repos = await init_db(temp_db_path, csv_path=None)
    # 'exposed' word with OLDER last_seen_at — must still lose to any 'learning' word
    await repos.stats.increment_exposed("old_exposed")
    # Manually backdate its last_seen_at to guarantee it's older
    await repos.stats._db.execute(
        "UPDATE vocab_stats SET last_seen_at = '2020-01-01 00:00:00' WHERE word = 'old_exposed'"
    )
    await repos.stats.increment_exposed("new_target", is_target=True)  # 'learning'
    result = await repos.stats.oldest_learning_word()
    assert result == "new_target"


@pytest.mark.asyncio
async def test_oldest_learning_word_falls_back_to_exposed_when_no_learning(temp_db_path):
    """Pool dry-up case: all learning words graduated, fall back to oldest
    exposed — this is the path that promotes a scaffolding word to target."""
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.increment_exposed("first_exposed")
    await repos.stats.increment_exposed("second_exposed")
    # Backdate 'first_exposed' to ensure it's older
    await repos.stats._db.execute(
        "UPDATE vocab_stats SET last_seen_at = '2020-01-01 00:00:00' WHERE word = 'first_exposed'"
    )
    result = await repos.stats.oldest_learning_word()
    assert result == "first_exposed"


@pytest.mark.asyncio
async def test_oldest_learning_word_ignores_mastered(temp_db_path):
    """Mastered words must never be returned as practice targets."""
    repos = await init_db(temp_db_path, csv_path=None)
    # Bump mastery to 3 via repeated +1 deltas with is_target=True
    await repos.stats.apply_delta("mastered_word", delta=1, exposed=True, is_target=True)
    await repos.stats.apply_delta("mastered_word", delta=1, is_target=True)
    await repos.stats.apply_delta("mastered_word", delta=1, is_target=True)
    assert (await repos.stats.get("mastered_word"))["status"] == "mastered"
    # Exposed word exists but is more recent — should still be returned because
    # mastered_word is filtered out.
    await repos.stats.increment_exposed("exposed_word")
    result = await repos.stats.oldest_learning_word()
    assert result == "exposed_word"
