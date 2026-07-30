# src/persistence/repos.py
"""Per-user Repo classes for the KET partner persistence layer.

Each Repo exposes a narrow async interface over a single table family.
Repos (Task 7) aggregates the 5 per-user Repos for one request.
"""
import aiosqlite

from src.persistence.models import WordRef  # MASTERY_CAP, derive_status imported when StatsRepo lands (Task 5)


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
