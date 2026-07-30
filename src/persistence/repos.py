# src/persistence/repos.py
"""Per-user Repo classes for the KET partner persistence layer.

Each Repo exposes a narrow async interface over a single table family.
Repos (Task 7) aggregates the 5 per-user Repos for one request.
"""
from datetime import datetime, timezone

import aiosqlite

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
