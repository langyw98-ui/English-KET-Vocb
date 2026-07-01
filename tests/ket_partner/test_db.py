import sqlite3
import pytest

from flow.ket_partner.db import init_db, VocabRepo, StatsRepo, ProfileRepo, LogRepo


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
    assert stats["status"] == "learning"


@pytest.mark.asyncio
async def test_stats_repo_apply_delta_floor_at_zero(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.apply_delta("cat", delta=-1, exposed=False)
    stats = await repos.stats.get("cat")
    assert stats["mastery_score"] == 0
    assert stats["wrong_count"] == 1


@pytest.mark.asyncio
async def test_status_transitions(temp_db_path):
    repos = await init_db(temp_db_path, csv_path=None)
    await repos.stats.apply_delta("cat", delta=1, exposed=True)
    assert (await repos.stats.get("cat"))["status"] == "learning"
    await repos.stats.apply_delta("cat", delta=1)
    assert (await repos.stats.get("cat"))["mastery_score"] == 2
    await repos.stats.apply_delta("cat", delta=1)
    assert (await repos.stats.get("cat"))["status"] == "mastered"
    await repos.stats.apply_delta("cat", delta=-1)
    assert (await repos.stats.get("cat"))["status"] == "learning"
    assert (await repos.stats.get("cat"))["mastery_score"] == 2


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
