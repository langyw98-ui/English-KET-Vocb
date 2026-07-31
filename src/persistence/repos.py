# src/persistence/repos.py
"""Per-user Repo classes for the KET partner persistence layer.

Each Repo exposes a narrow async interface over a single table family.
Repos (Task 7) aggregates the 5 per-user Repos for one request.
"""
import json
import re
from datetime import datetime, timezone

import aiosqlite

from flow.common import logger
from src.persistence.models import MASTERY_CAP, WordRef, derive_status


class VocabRepo:
    """ket_vocabulary / ket_vocab_topics tables — read access."""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None:
        self._db = db
        self._user_id = user_id

    async def get_topics_for_word(self, word: str, context: str = "") -> list[str]:
        async with self._db.execute(
            "SELECT topic FROM ket_vocab_topics "
            "WHERE word = ? AND context = ? ORDER BY topic",
            (word, context),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def get_ket_word(
        self, word: str, context: str = ""
    ) -> WordRef | None:
        async with self._db.execute(
            "SELECT word, context FROM ket_vocabulary "
            "WHERE word = ? COLLATE NOCASE AND context = ? LIMIT 1",
            (word, context),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return WordRef(word=row[0], context=row[1])

    async def get_ket_word_any_context(self, word: str) -> WordRef | None:
        async with self._db.execute(
            "SELECT word, context FROM ket_vocabulary "
            "WHERE word = ? COLLATE NOCASE "
            "ORDER BY (context = '') DESC, context ASC, "
            "(word = ? COLLATE BINARY) DESC, "
            "(word = lower(word)) DESC, word ASC LIMIT 1",
            (word, word),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return WordRef(word=row[0], context=row[1])

    async def words_in_topic_without_stats(self, topic: str) -> list[WordRef]:
        sql = (
            "SELECT v.word, v.context FROM ket_vocabulary v "
            "JOIN ket_vocab_topics t ON v.word = t.word AND v.context = t.context "
            "LEFT JOIN vocab_stats s ON v.word = s.word AND v.context = s.context AND s.user_id = ? "
            "WHERE t.topic = ? AND s.word IS NULL "
            "ORDER BY RANDOM() LIMIT 1"
        )
        async with self._db.execute(sql, (self._user_id, topic)) as cur:
            rows = await cur.fetchall()
        return [WordRef(word=r[0], context=r[1]) for r in rows]

    async def unexposed_notopic_words(self) -> list[WordRef]:
        sql = (
            "SELECT v.word, v.context FROM ket_vocabulary v "
            "WHERE NOT EXISTS ("
            "    SELECT 1 FROM ket_vocab_topics t "
            "    WHERE t.word = v.word AND t.context = v.context"
            ") AND NOT EXISTS ("
            "    SELECT 1 FROM vocab_stats s "
            "    WHERE s.word = v.word AND s.context = v.context AND s.user_id = ?"
            ") "
            "ORDER BY RANDOM() LIMIT 1"
        )
        async with self._db.execute(sql, (self._user_id,)) as cur:
            rows = await cur.fetchall()
        return [WordRef(word=r[0], context=r[1]) for r in rows]

    async def topics_with_unmastered(self, exclude: str | None = None) -> list[str]:
        sql = (
            "SELECT t.topic FROM ket_vocab_topics t "
            "LEFT JOIN vocab_stats s ON t.word = s.word AND t.context = s.context AND s.user_id = ? "
            "WHERE (s.status IS NULL OR s.status != 'mastered')"
        )
        params: list = [self._user_id]
        if exclude:
            sql += " AND t.topic != ?"
            params.append(exclude)
        sql += " GROUP BY t.topic ORDER BY RANDOM() LIMIT 1"
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def total_count(self) -> int:
        async with self._db.execute("SELECT COUNT(*) FROM ket_vocabulary") as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def seed_for_test(
        self,
        word: str,
        context: str = "",
        pos: str | None = None,
    ) -> None:
        """Test-only seed helper. Inserts a (word, context, pos) row into
        ket_vocabulary so the §4.4 orphan guard in StatsRepo.apply_delta
        admits subsequent stats writes for this sense.

        Production code MUST NOT call this — production vocab rows come from
        init_db / CSV import (see bootstrap._import_csv). Exposed as a small
        surface so tests don't reach into `_db.execute` directly.
        """
        await self._db.execute(
            "INSERT OR IGNORE INTO ket_vocabulary (word, context, pos, is_seed) "
            "VALUES (?, ?, ?, 0)",
            (word, context, pos),
        )
        await self._db.commit()


class StatsRepo:
    """vocab_stats table — read/write + mastery derivation.

    Option B deletes _category_where_sql/count_by_category/list_by_category;
    category rules now live in reporting/ket_partner/categories.py.
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> None:
        self._db = db
        self._user_id = user_id

    async def _vocab_has_default_sense(self, word: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM ket_vocabulary WHERE word = ? AND context = '' LIMIT 1",
            (word,),
        ) as cur:
            return await cur.fetchone() is not None

    async def get(self, word: str, context: str = "") -> dict | None:
        async with self._db.execute(
            "SELECT word, context, exposed_count, correct_count, wrong_count, "
            "mastery_score, status, first_seen_at, last_seen_at "
            "FROM vocab_stats WHERE user_id = ? AND word = ? AND context = ?",
            (self._user_id, word, context),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "word": row[0],
            "context": row[1],
            "exposed_count": row[2],
            "correct_count": row[3],
            "wrong_count": row[4],
            "mastery_score": row[5],
            "status": row[6],
            "first_seen_at": row[7],
            "last_seen_at": row[8],
        }

    async def apply_delta(
        self,
        word: str,
        context: str = "",
        delta: int = 0,
        exposed: bool = False,
        is_target: bool = False,
    ) -> dict | None:
        if context == "" and not await self._vocab_has_default_sense(word):
            return None

        existing = await self.get(word, context)
        now = datetime.now(timezone.utc)
        if existing is None:
            score = min(MASTERY_CAP, max(0, delta))
            await self._db.execute(
                "INSERT INTO vocab_stats "
                "(word, context, user_id, exposed_count, correct_count, wrong_count, "
                "mastery_score, status, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    word,
                    context,
                    self._user_id,
                    1 if exposed else 0,
                    1 if delta > 0 else 0,
                    1 if delta < 0 else 0,
                    score,
                    derive_status(None, score, is_target=is_target),
                    now,
                    now,
                ),
            )
        else:
            new_score = min(MASTERY_CAP, max(0, existing["mastery_score"] + delta))
            new_exposed = existing["exposed_count"] + (1 if exposed else 0)
            new_correct = existing["correct_count"] + (1 if delta > 0 else 0)
            new_wrong = existing["wrong_count"] + (1 if delta < 0 else 0)
            await self._db.execute(
                "UPDATE vocab_stats SET exposed_count=?, correct_count=?, "
                "wrong_count=?, mastery_score=?, status=?, last_seen_at=? "
                "WHERE user_id=? AND word=? AND context=?",
                (
                    new_exposed,
                    new_correct,
                    new_wrong,
                    new_score,
                    derive_status(existing["status"], new_score, is_target=is_target),
                    now,
                    self._user_id,
                    word,
                    context,
                ),
            )
        await self._db.commit()
        return await self.get(word, context)

    async def learning_count(self) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM vocab_stats WHERE user_id=? AND status='learning'",
            (self._user_id,),
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def oldest_learning_word(self) -> WordRef | None:
        async with self._db.execute(
            "SELECT word, context FROM vocab_stats WHERE user_id=? AND status='learning' "
            "ORDER BY last_seen_at ASC LIMIT 1",
            (self._user_id,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return WordRef(word=row[0], context=row[1])
        async with self._db.execute(
            "SELECT word, context FROM vocab_stats WHERE user_id=? AND status='exposed' "
            "ORDER BY last_seen_at ASC LIMIT 1",
            (self._user_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return WordRef(word=row[0], context=row[1])

    async def increment_exposed(
        self, word: str, context: str = "", is_target: bool = False
    ) -> None:
        await self.apply_delta(word, context=context, delta=0, exposed=True, is_target=is_target)

    async def list_all_with_vocab(self) -> list[dict]:
        """vocab_stats LEFT JOIN ket_vocabulary — all words including
        never-practiced. Replaces the old exporter's repos.stats._db.execute.
        """
        async with self._db.execute(
            "SELECT v.word, v.context, v.pos, "
            "COALESCE(s.exposed_count, 0) AS exposed_count, "
            "COALESCE(s.correct_count, 0) AS correct_count, "
            "COALESCE(s.wrong_count, 0) AS wrong_count, "
            "COALESCE(s.mastery_score, 0) AS mastery_score, "
            "COALESCE(s.status, 'new') AS status "
            "FROM ket_vocabulary v "
            "LEFT JOIN vocab_stats s ON v.word = s.word AND v.context = s.context "
            "AND s.user_id = ? "
            "ORDER BY v.word, v.context",
            (self._user_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "word": r[0],
                "context": r[1] or "",
                "pos": r[2],
                "exposed_count": r[3] or 0,
                "correct_count": r[4] or 0,
                "wrong_count": r[5] or 0,
                "mastery_score": r[6] or 0,
                "status": r[7] or "new",
            }
            for r in rows
        ]


class ProfileRepo:
    """kid_profile + users tables — user profile read/write."""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None:
        self._db = db
        self._user_id = user_id

    async def get(self) -> dict:
        async with self._db.execute(
            "SELECT u.nickname, u.age, p.total_turns, p.weakness_words, p.dialogue_strategy, "
            "p.in_refill_mode, p.last_new_word_turn, p.last_summary_turn, p.current_topic, p.updated_at "
            "FROM kid_profile p "
            "LEFT JOIN users u ON p.user_id = u.id "
            "WHERE p.user_id = ?",
            (self._user_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return {
                "nickname": "宝贝",
                "age": 8,
                "total_turns": 0,
                "weakness_words": [],
                "dialogue_strategy": None,
                "in_refill_mode": 0,
                "last_new_word_turn": 0,
                "last_summary_turn": 0,
                "current_topic": None,
                "updated_at": None,
            }
        return {
            "nickname": row[0] or "宝贝",
            "age": row[1] if row[1] is not None else 8,
            "total_turns": row[2] or 0,
            "weakness_words": json.loads(row[3]) if row[3] else [],
            "dialogue_strategy": row[4],
            "in_refill_mode": row[5] or 0,
            "last_new_word_turn": row[6] or 0,
            "last_summary_turn": row[7] or 0,
            "current_topic": row[8],
            "updated_at": row[9],
        }

    async def update(self, **fields) -> None:
        if not fields:
            logger.warning("ProfileRepo.update called with empty fields; no-op")
            return
        profile_allowed = {
            "total_turns",
            "weakness_words",
            "dialogue_strategy",
            "in_refill_mode",
            "last_new_word_turn",
            "last_summary_turn",
            "current_topic",
        }
        user_allowed = {"nickname", "age"}

        user_updates = {k: v for k, v in fields.items() if k in user_allowed}
        if user_updates:
            set_parts = [f"{k}=?" for k in user_updates]
            values = list(user_updates.values())
            values.append(self._user_id)
            await self._db.execute(
                f"UPDATE users SET {', '.join(set_parts)} WHERE id=?",
                values,
            )

        profile_updates = {k: v for k, v in fields.items() if k in profile_allowed}
        if profile_updates:
            set_parts = []
            values = []
            for k, v in profile_updates.items():
                if k == "weakness_words":
                    v = json.dumps(v, ensure_ascii=False)
                set_parts.append(f"{k}=?")
                values.append(v)
            set_parts.append("updated_at=?")
            values.append(datetime.now(timezone.utc))
            values.append(self._user_id)
            await self._db.execute(
                f"UPDATE kid_profile SET {', '.join(set_parts)} WHERE user_id=?",
                values,
            )
        await self._db.commit()


class LogRepo:
    """conversation_log table — chat history with turn linkage."""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None:
        self._db = db
        self._user_id = user_id

    async def append(
        self,
        role: str,
        content: str,
        words_used: list[str] | None = None,
        target_words: list[dict[str, str]] | None = None,
        turn_id: int | None = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO conversation_log (user_id, role, content, words_used, target_words, turn_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._user_id,
                role,
                content,
                json.dumps(words_used or [], ensure_ascii=False),
                json.dumps(target_words or [], ensure_ascii=False),
                turn_id,
            ),
        )
        await self._db.commit()

    async def recent(self, limit: int = 5) -> list[dict]:
        async with self._db.execute(
            "SELECT role, content, words_used, target_words, turn_id, created_at "
            "FROM conversation_log WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (self._user_id, limit),
        ) as cur:
            rows = list(await cur.fetchall())
        rows.reverse()
        return [
            {
                "role": r[0],
                "content": r[1],
                "words_used": json.loads(r[2]) if r[2] else [],
                "target_words": json.loads(r[3]) if r[3] else [],
                "turn_id": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    async def append_session_start(self) -> None:
        await self.append("system", "session_start", words_used=[], target_words=[])

    async def last_ai_message(self) -> dict | None:
        sql = (
            "SELECT content, words_used, target_words FROM conversation_log "
            "WHERE role='ai' AND user_id=? AND id > COALESCE("
            "    (SELECT MAX(id) FROM conversation_log WHERE role='system' AND content='session_start' AND user_id=?), 0"
            ") ORDER BY id DESC LIMIT 1"
        )
        async with self._db.execute(sql, (self._user_id, self._user_id)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "content": row[0],
            "words_used": json.loads(row[1]) if row[1] else [],
            "target_words": json.loads(row[2]) if row[2] else [],
        }


class RecentSentencesRepo:
    """recent_sentences table — read/write."""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None:
        self._db = db
        self._user_id = user_id

    async def list_recent(self, limit: int = 20) -> list[str]:
        async with self._db.execute(
            "SELECT sentence FROM recent_sentences WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (self._user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def append(self, sentence: str, window: int = 20) -> None:
        now = datetime.now(timezone.utc)
        await self._db.execute(
            "INSERT INTO recent_sentences (user_id, sentence, created_at) VALUES (?, ?, ?)",
            (self._user_id, sentence, now),
        )
        await self._db.execute(
            "DELETE FROM recent_sentences WHERE user_id=? AND rowid NOT IN ("
            "    SELECT rowid FROM recent_sentences WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?"
            ")",
            (self._user_id, self._user_id, window),
        )
        await self._db.commit()

    async def list_recent_scaffolding(self, window: int = 20) -> list[list[str]]:
        sentences = await self.list_recent(limit=window)
        scaffolding_list: list[list[str]] = []
        pattern = re.compile(r"[A-Za-z']+")
        for s in sentences:
            tokens = [t.lower() for t in pattern.findall(s)]
            scaffolding_list.append(tokens)
        return scaffolding_list


class Repos:
    """Facade over 5 per-user Repos. Constructed once per request; user_id isolates."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> None:
        self._db = db
        self._user_id = user_id
        self.vocab = VocabRepo(db, user_id)
        self.stats = StatsRepo(db, user_id)
        self.profile = ProfileRepo(db, user_id)
        self.log = LogRepo(db, user_id)
        self.recent = RecentSentencesRepo(db, user_id)

    @classmethod
    def for_user(
        cls,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> "Repos":
        return cls(db, user_id)

    async def close(self) -> None:
        await self._db.close()
