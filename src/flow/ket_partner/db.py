import csv
import json
import re
import sqlite3
from datetime import datetime
from os.path import dirname, join
from typing import Dict, List, NamedTuple, Optional, TYPE_CHECKING

import aiosqlite

from flow.common import logger

if TYPE_CHECKING:
    from flow.ket_partner.config import KetConfig


class WordRef(NamedTuple):
    """A (word, context) pair — the unit of practice.

    Threads through vocab_selector → agent → evaluator so target context
    reaches every prompt that needs it. Spec §4.1.
    """
    word: str
    context: str = ""


# Mastery ceiling. A score of 2 graduates a word to 'mastered'; without a
# hard cap, repeated correct translations keep accumulating (3, 4, 5, ...)
# and the kid must burn many wrong answers before a previously-mastered word
# demotes back into the learning pool. Capping at 2 keeps the demotion path
# short: 2 → 1 ('learning') in a single wrong answer — so a kid who once
# knew a word but has drifted is re-detected quickly and the word re-enters
# active practice.
MASTERY_CAP = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ket_vocabulary (
    word    TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    pos     TEXT,
    is_seed INTEGER DEFAULT 0,
    PRIMARY KEY (word, context)
);
CREATE INDEX IF NOT EXISTS idx_vocab_seed ON ket_vocabulary(is_seed);

CREATE TABLE IF NOT EXISTS ket_vocab_topics (
    word    TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    topic   TEXT NOT NULL,
    PRIMARY KEY (word, context, topic)
);
CREATE INDEX IF NOT EXISTS idx_vocab_topics_topic ON ket_vocab_topics(topic);
CREATE INDEX IF NOT EXISTS idx_vocab_topics_lookup ON ket_vocab_topics(word, context);

CREATE TABLE IF NOT EXISTS vocab_stats (
    word          TEXT NOT NULL,
    context       TEXT NOT NULL DEFAULT '',
    user_id       TEXT NOT NULL DEFAULT 'default',
    exposed_count INTEGER DEFAULT 0,
    correct_count INTEGER DEFAULT 0,
    wrong_count   INTEGER DEFAULT 0,
    mastery_score INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'new',
    first_seen_at TIMESTAMP,
    last_seen_at  TIMESTAMP,
    PRIMARY KEY (user_id, word, context)
);
CREATE INDEX IF NOT EXISTS idx_stats_user_status ON vocab_stats(user_id, status);

CREATE TABLE IF NOT EXISTS conversation_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL DEFAULT 'default',
    role         TEXT,
    content      TEXT,
    words_used   TEXT,
    target_words TEXT,
    turn_id      INTEGER,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_log_user_id ON conversation_log(user_id, id);

CREATE TABLE IF NOT EXISTS kid_profile (
    user_id            TEXT PRIMARY KEY,
    total_turns        INTEGER DEFAULT 0,
    weakness_words     TEXT,
    dialogue_strategy  TEXT,
    in_refill_mode     INTEGER DEFAULT 0,
    last_new_word_turn INTEGER DEFAULT 0,
    last_summary_turn  INTEGER DEFAULT 0,
    current_topic      TEXT,
    updated_at         TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    nickname    TEXT NOT NULL,
    age         INTEGER NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recent_sentences (
    user_id     TEXT NOT NULL,
    sentence    TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_recent_user_created ON recent_sentences(user_id, created_at DESC);
"""


def _derive_status(
    current_status: Optional[str],
    mastery_score: int,
    is_target: bool = False,
) -> str:
    """Derive vocab_stats.status from inputs."""
    if mastery_score >= MASTERY_CAP:
        return "mastered"
    if current_status == "mastered":
        return "learning" if mastery_score <= 1 else "mastered"
    if is_target:
        return "learning"
    if current_status is None:
        return "exposed"
    return current_status


class VocabRepo:
    def __init__(self, db: aiosqlite.Connection, user_id: str = "default"):
        self._db = db
        self._user_id = user_id

    async def get_topics_for_word(self, word: str, context: str = "") -> List[str]:
        async with self._db.execute(
            "SELECT topic FROM ket_vocab_topics "
            "WHERE word = ? AND context = ? ORDER BY topic",
            (word, context),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def get_ket_word(
        self, word: str, context: str = ""
    ) -> Optional[WordRef]:
        async with self._db.execute(
            "SELECT word, context FROM ket_vocabulary "
            "WHERE word = ? COLLATE NOCASE AND context = ? LIMIT 1",
            (word, context),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return WordRef(word=row[0], context=row[1])

    async def get_ket_word_any_context(self, word: str) -> Optional[WordRef]:
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

    async def words_in_topic_without_stats(self, topic: str) -> List[WordRef]:
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

    async def unexposed_notopic_words(self) -> List[WordRef]:
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

    async def topics_with_unmastered(self, exclude: Optional[str] = None) -> List[str]:
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
    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
        config: Optional["KetConfig"] = None,
    ):
        self._db = db
        self._user_id = user_id
        if config is None:
            from flow.ket_partner.config import load_config
            config = load_config()
        self._config = config

    async def _vocab_has_default_sense(self, word: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM ket_vocabulary WHERE word = ? AND context = '' LIMIT 1",
            (word,),
        ) as cur:
            return await cur.fetchone() is not None

    async def get(self, word: str, context: str = "") -> Optional[dict]:
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
    ) -> Optional[dict]:
        if context == "" and not await self._vocab_has_default_sense(word):
            return None

        existing = await self.get(word, context)
        now = datetime.utcnow()
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
                    _derive_status(None, score, is_target=is_target),
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
                    _derive_status(existing["status"], new_score, is_target=is_target),
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

    async def oldest_learning_word(self) -> Optional[WordRef]:
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

    async def count_by_category(self, category: str) -> int:
        sql, params = self._category_where_sql(category)
        async with self._db.execute(f"SELECT COUNT(*) FROM ({sql})", params) as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def list_by_category(
        self, category: str, offset: int = 0, limit: int = 100
    ) -> List[dict]:
        sql, params = self._category_where_sql(category)
        sql += " LIMIT ? OFFSET ?"
        full_params = (*params, limit, offset)
        async with self._db.execute(sql, full_params) as cur:
            rows = await cur.fetchall()
        return [
            {
                "word": r[0],
                "context": r[1] or "",
                "mastery_score": r[2] if len(r) > 2 else 0,
                "exposed_count": r[3] if len(r) > 3 else 0,
                "correct_count": r[4] if len(r) > 4 else 0,
                "wrong_count": r[5] if len(r) > 5 else 0,
                "status": r[6] if len(r) > 6 else "new",
            }
            for r in rows
        ]

    def _category_where_sql(self, category: str) -> tuple[str, tuple]:
        wc_min = self._config.struggling_threshold.wrong_count_min
        ec_min = self._config.struggling_threshold.exposed_count_min

        if category == "mastered":
            return (
                "SELECT word, context, mastery_score, exposed_count, correct_count, wrong_count, status "
                "FROM vocab_stats WHERE user_id=? AND status='mastered'",
                (self._user_id,),
            )
        if category == "learning":
            return (
                "SELECT word, context, mastery_score, exposed_count, correct_count, wrong_count, status "
                "FROM vocab_stats WHERE user_id=? AND status='learning'",
                (self._user_id,),
            )
        if category == "struggling":
            return (
                "SELECT word, context, mastery_score, exposed_count, correct_count, wrong_count, status "
                "FROM vocab_stats WHERE user_id=? "
                "AND status NOT IN ('mastered', 'learning') "
                "AND exposed_count > 0 "
                "AND (wrong_count >= ? OR (exposed_count >= ? AND mastery_score = 0))",
                (self._user_id, wc_min, ec_min),
            )
        if category == "used":
            return (
                "SELECT word, context, mastery_score, exposed_count, correct_count, wrong_count, status "
                "FROM vocab_stats WHERE user_id=? "
                "AND exposed_count > 0 "
                "AND status NOT IN ('mastered', 'learning') "
                "AND NOT (wrong_count >= ? OR (exposed_count >= ? AND mastery_score = 0))",
                (self._user_id, wc_min, ec_min),
            )
        if category == "unused":
            return (
                "SELECT v.word, v.context, COALESCE(s.mastery_score, 0) AS mastery_score, "
                "COALESCE(s.exposed_count, 0) AS exposed_count, "
                "COALESCE(s.correct_count, 0) AS correct_count, "
                "COALESCE(s.wrong_count, 0) AS wrong_count, "
                "COALESCE(s.status, 'new') AS status "
                "FROM ket_vocabulary v "
                "LEFT JOIN vocab_stats s ON s.word = v.word AND s.context = v.context AND s.user_id = ? "
                "WHERE s.word IS NULL OR s.exposed_count = 0",
                (self._user_id,),
            )
        raise ValueError(f"invalid category: {category}")


class ProfileRepo:
    def __init__(self, db: aiosqlite.Connection, user_id: str = "default"):
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
            return
        profile_allowed = {
            "total_turns", "weakness_words", "dialogue_strategy",
            "in_refill_mode", "last_new_word_turn", "last_summary_turn",
            "current_topic",
        }
        user_allowed = {"nickname", "age"}

        user_updates = {k: v for k, v in fields.items() if k in user_allowed}
        if user_updates:
            set_parts = [f"{k}=?" for k in user_updates.keys()]
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
            values.append(datetime.utcnow())
            values.append(self._user_id)
            await self._db.execute(
                f"UPDATE kid_profile SET {', '.join(set_parts)} WHERE user_id=?",
                values,
            )
        await self._db.commit()


class LogRepo:
    def __init__(self, db: aiosqlite.Connection, user_id: str = "default"):
        self._db = db
        self._user_id = user_id

    async def append(
        self,
        role: str,
        content: str,
        words_used: Optional[List[str]] = None,
        target_words: Optional[List[Dict[str, str]]] = None,
        turn_id: Optional[int] = None,
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

    async def recent(self, limit: int = 5) -> List[dict]:
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

    async def last_ai_message(self) -> Optional[dict]:
        async with self._db.execute(
            "SELECT content, words_used, target_words FROM conversation_log "
            "WHERE role='ai' AND user_id=? ORDER BY id DESC LIMIT 1",
            (self._user_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "content": row[0],
            "words_used": json.loads(row[1]) if row[1] else [],
            "target_words": json.loads(row[2]) if row[2] else [],
        }


class RecentSentencesRepo:
    def __init__(self, db: aiosqlite.Connection, user_id: str = "default"):
        self._db = db
        self._user_id = user_id

    async def list_recent(self, limit: int = 20) -> List[str]:
        async with self._db.execute(
            "SELECT sentence FROM recent_sentences WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (self._user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def append(self, sentence: str, window: int = 20) -> None:
        now = datetime.utcnow()
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

    async def list_recent_scaffolding(self, window: int = 20) -> List[List[str]]:
        sentences = await self.list_recent(limit=window)
        scaffolding_list: List[List[str]] = []
        pattern = re.compile(r"[A-Za-z']+")
        for s in sentences:
            tokens = [t.lower() for t in pattern.findall(s)]
            scaffolding_list.append(tokens)
        return scaffolding_list


class Repos:
    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
        config: Optional["KetConfig"] = None,
    ):
        self._db = db
        self._user_id = user_id
        if config is None:
            from flow.ket_partner.config import load_config
            config = load_config()
        self._config = config
        self.vocab = VocabRepo(db, user_id)
        self.stats = StatsRepo(db, user_id, self._config)
        self.profile = ProfileRepo(db, user_id)
        self.log = LogRepo(db, user_id)
        self.recent = RecentSentencesRepo(db, user_id)

    @classmethod
    def for_user(
        cls,
        db: aiosqlite.Connection,
        user_id: str = "default",
        config: Optional["KetConfig"] = None,
    ) -> "Repos":
        return cls(db, user_id, config)

    async def close(self):
        await self._db.close()


_DEFAULT_CSV = join(dirname(__file__), "..", "..", "..", "data", "KET_vocabulary.csv")


async def init_db(
    db_path: str,
    csv_path: Optional[str] = None,
    default_nickname: str = "宝贝",
    default_age: int = 8,
) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.executescript(_SCHEMA)

    await db.execute(
        "INSERT OR IGNORE INTO users (id, nickname, age) VALUES ('default', ?, ?)",
        (default_nickname, default_age),
    )
    await db.execute(
        "INSERT OR IGNORE INTO kid_profile (user_id, total_turns) VALUES ('default', 0)",
    )
    await db.commit()

    if csv_path:
        await _import_csv(db, csv_path)

    return db


async def _import_csv(db: aiosqlite.Connection, csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            word = (row.get("word") or "").strip()
            pos = (row.get("part_of_speech") or "").strip()
            topic_raw = (row.get("topic") or "").strip()
            context = (row.get("context") or "").strip()
            if not word:
                continue
            topics = [t.strip() for t in topic_raw.split(";") if t.strip()]
            await db.execute(
                "INSERT OR IGNORE INTO ket_vocabulary (word, context, pos, is_seed) "
                "VALUES (?, ?, ?, 0)",
                (word, context, pos),
            )
            for t in topics:
                await db.execute(
                    "INSERT OR IGNORE INTO ket_vocab_topics (word, context, topic) "
                    "VALUES (?, ?, ?)",
                    (word, context, t),
                )
            count += 1
        await db.commit()
        logger.info(f"Imported {count} rows from {csv_path}")
