# tests/persistence/test_repos.py
import tempfile

import pytest

from src.persistence.bootstrap import init_db
from src.persistence.models import WordRef
from src.persistence.repos import (
    LogRepo,
    ProfileRepo,
    Repos,
    StatsRepo,
    VocabRepo,
)


async def _init_vocab_repo(db_path: str, csv_path: str | None = None) -> VocabRepo:
    """Open DB via bootstrap.init_db, wrap a VocabRepo around it.

    Each test owns the connection and must close it in a finally block,
    OR rely on the temp_db_path fixture's cleanup (file-based DB). Tests
    that need stats/log/recent access should NOT use this helper — they
    belong to later tasks.
    """
    db = await init_db(db_path, csv_path=csv_path)
    return VocabRepo(db, "default")


async def _init_stats_repo(db_path: str, csv_path: str | None = None) -> StatsRepo:
    """Open DB via bootstrap.init_db, wrap a StatsRepo around it.

    Same ownership contract as _init_vocab_repo: caller owns the connection.
    """
    db = await init_db(db_path, csv_path=csv_path)
    return StatsRepo(db, "default")


async def _init_profile_repo(db_path: str) -> ProfileRepo:
    """Open DB via bootstrap.init_db, wrap a ProfileRepo around it.

    Same ownership contract as _init_vocab_repo: caller owns the connection.
    """
    db = await init_db(db_path, csv_path=None)
    return ProfileRepo(db, "default")


async def _init_log_repo(db_path: str) -> LogRepo:
    """Open DB via bootstrap.init_db, wrap a LogRepo around it.

    Same ownership contract as _init_vocab_repo: caller owns the connection.
    """
    db = await init_db(db_path, csv_path=None)
    return LogRepo(db, "default")


@pytest.mark.asyncio
class TestVocabRepo:
    async def test_get_ket_word_returns_wordref_with_context(self, temp_db_path):
        """Spec §4.1: get_ket_word returns a WordRef carrying both word and
        context so callers don't have to thread context separately."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "smart,adj,,clever\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        wr = await repo.get_ket_word("smart", context="clever")
        assert wr == WordRef(word="smart", context="clever")

    async def test_get_ket_word_returns_none_when_context_mismatch(self, temp_db_path):
        """Spec §5.1: precise lookup. smart has only (smart, clever); asking
        for (smart, '') returns None — this is the property the validator
        MUST avoid by using get_ket_word_any_context."""
        csv_text = "word,part_of_speech,topic,context\nsmart,adj,,clever\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        assert await repo.get_ket_word("smart", context="") is None

    async def test_get_ket_word_any_context_prefers_default(self, temp_db_path):
        """Spec §5.1: when (word, '') exists, return it; otherwise return the
        first by context ASC. Preferring default keeps the any-context lookup
        stable across schema growth."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "design,v,,\n"            # default sense
            "design,n,,planning\n"
            "design,n,,drawing\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        wr = await repo.get_ket_word_any_context("design")
        assert wr == WordRef(word="design", context="")

    async def test_get_ket_word_any_context_returns_first_when_no_default(self, temp_db_path):
        """Spec §5.1: 7 orphan-skip words have no (word, '') row. The
        any-context lookup must still recognize them as KET — return the
        lexicographically first context."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "smart,adj,,stylish\n"
            "smart,adj,,clever\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        wr = await repo.get_ket_word_any_context("smart")
        assert wr == WordRef(word="smart", context="clever")

    async def test_words_in_topic_without_stats_returns_wordref(self, temp_db_path):
        """Spec §6: vocab_selector transparently threads WordRef through; the
        underlying query must return (word, context) tuples.

        Uses a single-candidate CSV because the SQL is ORDER BY RANDOM() LIMIT 1
        (production indexes [0]) -- deterministic assertion needs exactly one
        candidate."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "cat,n,Animals,\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        candidates = await repo.words_in_topic_without_stats("Animals")
        assert candidates == [WordRef(word="cat", context="")]


@pytest.mark.asyncio
class TestStatsRepo:
    async def test_stats_repo_apply_delta_creates_row(self, temp_db_path):
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.apply_delta("cat", delta=1, exposed=True)
        stats = await repo.get("cat")
        assert stats["mastery_score"] == 1
        assert stats["exposed_count"] == 1
        # No is_target → not a target word → 'exposed' (passive exposure with a correct answer)
        assert stats["status"] == "exposed"

    async def test_stats_repo_apply_delta_floor_at_zero(self, temp_db_path):
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.apply_delta("cat", delta=-1, exposed=False)
        stats = await repo.get("cat")
        assert stats["mastery_score"] == 0
        assert stats["wrong_count"] == 1

    async def test_stats_repo_apply_delta_caps_mastery_at_cap(self, temp_db_path):
        """Repeated +1 deltas beyond CAP must stick at CAP=2. Without the cap,
        the score accumulates indefinitely and the kid has to burn many wrong
        answers to demote a previously-mastered word back into the learning
        pool. Capping at 2 keeps the demotion path short: a single wrong answer
        (2→1) demotes immediately."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        for _ in range(6):
            await repo.apply_delta("cat", delta=1, exposed=True)
        stats = await repo.get("cat")
        assert stats["mastery_score"] == 2, (
            f"mastery_score must cap at 2 (got {stats['mastery_score']})"
        )
        assert stats["status"] == "mastered"
        # exposed_count and correct_count still accumulate — the cap is on
        # mastery_score only, not the audit counters.
        assert stats["exposed_count"] == 6
        assert stats["correct_count"] == 6

    async def test_stats_repo_apply_delta_caps_single_large_delta(self, temp_db_path):
        """A single delta > CAP (e.g., backfill or test setup) must clamp at CAP=2."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.apply_delta("cat", delta=5, exposed=True)
        stats = await repo.get("cat")
        assert stats["mastery_score"] == 2

    async def test_status_transitions(self, temp_db_path):
        """A scaffolding-only word: stays 'exposed' below mastery CAP=2,
        graduates to 'mastered' at 2, demotes to 'learning' immediately on a
        single wrong answer (no absorption buffer with CAP=2)."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        # delta=1 exposed=True, no is_target → 'exposed' (passive + correct)
        await repo.apply_delta("cat", delta=1, exposed=True)
        assert (await repo.get("cat"))["status"] == "exposed"
        assert (await repo.get("cat"))["mastery_score"] == 1
        # 1→2: crosses cap → mastered
        await repo.apply_delta("cat", delta=1)
        assert (await repo.get("cat"))["mastery_score"] == 2
        assert (await repo.get("cat"))["status"] == "mastered"
        # 2→1: no absorption buffer with CAP=2 — demotes immediately to learning
        await repo.apply_delta("cat", delta=-1)
        assert (await repo.get("cat"))["mastery_score"] == 1
        assert (await repo.get("cat"))["status"] == "learning"

    async def test_increment_exposed_creates_exposed_row_for_scaffolding(self, temp_db_path):
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.increment_exposed("cat")
        stats = await repo.get("cat")
        assert stats["exposed_count"] == 1
        assert stats["mastery_score"] == 0
        assert stats["status"] == "exposed"

    async def test_increment_exposed_creates_learning_row_for_target(self, temp_db_path):
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.increment_exposed("cat", is_target=True)
        stats = await repo.get("cat")
        assert stats["exposed_count"] == 1
        assert stats["status"] == "learning"

    async def test_increment_exposed_promotes_existing_exposed_to_learning(self, temp_db_path):
        """When a word previously seen as scaffolding becomes a target, its
        status must promote from 'exposed' to 'learning'."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.increment_exposed("cat")  # scaffolding exposure
        assert (await repo.get("cat"))["status"] == "exposed"
        await repo.increment_exposed("cat", is_target=True)  # now target
        assert (await repo.get("cat"))["status"] == "learning"
        assert (await repo.get("cat"))["exposed_count"] == 2

    async def test_increment_exposed_preserves_learning_on_scaffolding_reexposure(self, temp_db_path):
        """A word that was target once, then appears as scaffolding in a later
        sentence, must stay 'learning' (target history is sticky)."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('cat', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.increment_exposed("cat", is_target=True)
        await repo.increment_exposed("cat")  # scaffolding re-exposure
        assert (await repo.get("cat"))["status"] == "learning"
        assert (await repo.get("cat"))["exposed_count"] == 2

    async def test_oldest_learning_word_returns_none_when_empty(self, temp_db_path):
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        assert await repo.oldest_learning_word() is None

    async def test_oldest_learning_word_prefers_learning_over_exposed(self, temp_db_path):
        """When both pools exist, learning wins — exposed is only the fallback."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('old_exposed', '', 'n', 0)"
        )
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('new_target', '', 'n', 0)"
        )
        await repo._db.commit()
        # 'exposed' word with OLDER last_seen_at — must still lose to any 'learning' word
        await repo.increment_exposed("old_exposed")
        # Manually backdate its last_seen_at to guarantee it's older
        await repo._db.execute(
            "UPDATE vocab_stats SET last_seen_at = '2020-01-01 00:00:00' WHERE word = 'old_exposed' AND context = ''"
        )
        await repo.increment_exposed("new_target", is_target=True)  # 'learning'
        result = await repo.oldest_learning_word()
        assert result == WordRef(word="new_target", context="")

    async def test_oldest_learning_word_falls_back_to_exposed_when_no_learning(self, temp_db_path):
        """Pool dry-up case: all learning words graduated, fall back to oldest
        exposed — this is the path that promotes a scaffolding word to target."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('first_exposed', '', 'n', 0)"
        )
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('second_exposed', '', 'n', 0)"
        )
        await repo._db.commit()
        await repo.increment_exposed("first_exposed")
        await repo.increment_exposed("second_exposed")
        # Backdate 'first_exposed' to ensure it's older
        await repo._db.execute(
            "UPDATE vocab_stats SET last_seen_at = '2020-01-01 00:00:00' WHERE word = 'first_exposed' AND context = ''"
        )
        result = await repo.oldest_learning_word()
        assert result == WordRef(word="first_exposed", context="")

    async def test_oldest_learning_word_ignores_mastered(self, temp_db_path):
        """Mastered words must never be returned as practice targets."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('mastered_word', '', 'n', 0)"
        )
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES ('exposed_word', '', 'n', 0)"
        )
        await repo._db.commit()
        # Bump mastery to 3 via repeated +1 deltas with is_target=True
        await repo.apply_delta("mastered_word", delta=1, exposed=True, is_target=True)
        await repo.apply_delta("mastered_word", delta=1, is_target=True)
        await repo.apply_delta("mastered_word", delta=1, is_target=True)
        assert (await repo.get("mastered_word"))["status"] == "mastered"
        # Exposed word exists but is more recent — should still be returned because
        # mastered_word is filtered out.
        await repo.increment_exposed("exposed_word")
        result = await repo.oldest_learning_word()
        assert result == WordRef(word="exposed_word", context="")

    async def test_oldest_learning_word_returns_wordref(self, temp_db_path):
        """Spec §5.2: oldest_learning_word returns WordRef so vocab_selector
        can thread the context through to the agent."""
        csv_text = "word,part_of_speech,topic,context\ncat,n,Animals,\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_stats_repo(temp_db_path, csv_path=csv_path)
        await repo.increment_exposed("cat", is_target=True)
        result = await repo.oldest_learning_word()
        assert result == WordRef(word="cat", context="")

    async def test_apply_delta_skips_orphan_default_sense(self, temp_db_path):
        """Spec §4.4: smart has only (smart, clever) in vocab. Writing stats
        at (smart, '') must silently no-op — otherwise /exportstats shows a
        KET row that the vocab table doesn't have."""
        csv_text = "word,part_of_speech,topic,context\nsmart,adj,,clever\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_stats_repo(temp_db_path, csv_path=csv_path)
        # Scaffolding-style call: context="" default.
        result = await repo.apply_delta("smart", delta=1, exposed=True)
        assert result is None
        # Confirm no stats row was created.
        assert await repo.get("smart") is None
        assert await repo.get("smart", context="clever") is None

    async def test_apply_delta_writes_default_sense_when_vocab_has_row(self, temp_db_path):
        """Spec §15: design/follow/share/train all have a (word, '') row in
        vocab. For these words, scaffolding exposure DOES write to (word, '')."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "design,v,,\n"
            "design,n,,planning\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_stats_repo(temp_db_path, csv_path=csv_path)
        await repo.apply_delta("design", delta=1, exposed=True)
        default = await repo.get("design")
        assert default is not None
        assert default["exposed_count"] == 1
        # Specific sense row stays untouched.
        planning = await repo.get("design", context="planning")
        assert planning is None

    async def test_apply_delta_threads_context_for_target_path(self, temp_db_path):
        """Spec §4.3: target path passes the real context through. apply_delta
        writes to (word, context) without consulting the orphan guard
        (non-empty context is always valid by source)."""
        csv_text = "word,part_of_speech,topic,context\nsmart,adj,,clever\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_stats_repo(temp_db_path, csv_path=csv_path)
        await repo.apply_delta("smart", context="clever", delta=1, is_target=True)
        row = await repo.get("smart", context="clever")
        assert row is not None
        assert row["mastery_score"] == 1
        assert row["status"] == "learning"

    async def test_list_all_with_vocab_returns_all_words(self, temp_db_path):
        """list_all_with_vocab returns every ket_vocabulary row LEFT JOINed
        with vocab_stats — including words the kid never practiced. Replaces
        the old exporter's repos.stats._db.execute private access."""
        repo = await _init_stats_repo(temp_db_path, csv_path=None)
        # Manually insert two vocab rows; no stats yet
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES (?, ?, ?, 0)",
            ("cat", "", "n"),
        )
        await repo._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES (?, ?, ?, 0)",
            ("dog", "", "n"),
        )
        await repo.apply_delta("cat", delta=1, exposed=True)
        await repo._db.commit()

        rows = await repo.list_all_with_vocab()
        words = {r["word"] for r in rows}
        assert words == {"cat", "dog"}
        cat_row = next(r for r in rows if r["word"] == "cat")
        assert cat_row["exposed_count"] == 1
        assert cat_row["correct_count"] == 1
        dog_row = next(r for r in rows if r["word"] == "dog")
        assert dog_row["exposed_count"] == 0
        assert dog_row["status"] == "new"


@pytest.mark.asyncio
class TestProfileRepo:
    async def test_profile_repo_init_creates_single_row(self, temp_db_path):
        """ProfileRepo.get() returns a default profile when the database is empty.
        Verifies that a fresh DB starts with zeroed counters and None topic."""
        repo = await _init_profile_repo(temp_db_path)
        profile = await repo.get()
        assert profile["total_turns"] == 0
        assert profile["in_refill_mode"] == 0
        assert profile["current_topic"] is None


@pytest.mark.asyncio
class TestLogRepo:
    async def test_last_ai_message_returns_none_when_no_ai_rows(self, temp_db_path):
        """LogRepo.last_ai_message() returns None when no AI messages exist."""
        repo = await _init_log_repo(temp_db_path)
        assert await repo.last_ai_message() is None

    async def test_last_ai_message_returns_latest_when_no_session_start(self, temp_db_path):
        """Backward compat: databases populated before session_start was added
        must still return the latest AI row when no marker exists."""
        repo = await _init_log_repo(temp_db_path)
        await repo.append("ai", "old sentence", words_used=["old"], turn_id=1)
        await repo.append("ai", "newer sentence", words_used=["new"], turn_id=2)
        result = await repo.last_ai_message()
        assert result is not None
        assert result["content"] == "newer sentence"

    async def test_log_append_persists_target_words_with_context(self, temp_db_path):
        """Spec §5.3: target_words is now List[{'word':..., 'context':...}]
        so the cross-turn rehydration in init_state can recover both fields."""
        repo = await _init_log_repo(temp_db_path)
        await repo.append(
            "ai",
            "The smart kid.",
            words_used=["the", "smart", "kid"],
            target_words=[{"word": "smart", "context": "clever"}],
            turn_id=1,
        )
        last = await repo.last_ai_message()
        assert last is not None
        assert last["target_words"] == [{"word": "smart", "context": "clever"}]


async def _init_repos(db_path: str, csv_path: str | None = None) -> Repos:
    """Aggregate helper for tests that exercise cross-repo behavior.

    Opens DB via bootstrap.init_db, wraps a full Repos facade. Caller owns
    the connection (Repos.close is the caller's responsibility, or rely on
    temp_db_path file cleanup).
    """
    db = await init_db(db_path, csv_path=csv_path)
    return Repos.for_user(db, "default")


@pytest.mark.asyncio
class TestRecentSentencesRepo:
    async def test_recent_sentences_repo(self, temp_db_path):
        db = await init_db(temp_db_path, csv_path=None)
        try:
            repos_a = Repos.for_user(db, "user_a")
            repos_b = Repos.for_user(db, "user_b")

            await repos_a.recent.append("The cat slept on the mat.")
            recent_a = await repos_a.recent.list_recent(limit=10)
            recent_b = await repos_b.recent.list_recent(limit=10)

            assert recent_a == ["The cat slept on the mat."]
            assert recent_b == []

            scaffolding_a = await repos_a.recent.list_recent_scaffolding(window=20)
            assert scaffolding_a == [["the", "cat", "slept", "on", "the", "mat"]]
        finally:
            await db.close()


@pytest.mark.asyncio
class TestRepos:
    async def test_multi_user_isolation(self, temp_db_path):
        db = await init_db(temp_db_path, csv_path=None)
        try:
            repos_a = Repos.for_user(db, "user_a")
            repos_b = Repos.for_user(db, "user_b")

            await repos_a.stats.increment_exposed("cat", context="slipping")

            stats_a = await repos_a.stats.get("cat", context="slipping")
            stats_b = await repos_b.stats.get("cat", context="slipping")

            assert stats_a["exposed_count"] == 1
            assert stats_b is None
        finally:
            await db.close()

    async def test_ket_vocabulary_uses_composite_pk(self, temp_db_path):
        """Spec §2.1: ket_vocabulary PK is (word, context). Verifying via PRAGMA
        because a wrong PK silently breaks INSERT OR IGNORE uniqueness."""
        db = await init_db(temp_db_path, csv_path=None)
        async with db.execute("PRAGMA table_info(ket_vocabulary)") as cur:
            rows = await cur.fetchall()
        pk_cols = {r[1] for r in rows if r[5] != 0}   # r[5] is pk flag
        assert pk_cols == {"word", "context"}

    async def test_for_user_no_config_param(self, temp_db_path):
        """Spec §3: Repos.for_user signature drops config (Option B).
        Constructing without config must succeed — no KetConfig import."""
        db = await init_db(temp_db_path, csv_path=None)
        repos = Repos.for_user(db, "default")
        assert repos.vocab is not None
        assert repos.stats is not None
        await repos.close()
