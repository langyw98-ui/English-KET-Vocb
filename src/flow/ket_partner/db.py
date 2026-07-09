import csv
import json
import sqlite3
from datetime import datetime
from os.path import dirname, join
from typing import List, Optional

import aiosqlite

from flow.common import logger

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
    exposed_count INTEGER DEFAULT 0,
    correct_count INTEGER DEFAULT 0,
    wrong_count   INTEGER DEFAULT 0,
    mastery_score INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'new',
    first_seen_at TIMESTAMP,
    last_seen_at  TIMESTAMP,
    PRIMARY KEY (word, context)
);
CREATE INDEX IF NOT EXISTS idx_stats_status ON vocab_stats(status);

CREATE TABLE IF NOT EXISTS conversation_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    role         TEXT,
    content      TEXT,
    words_used   TEXT,
    target_words TEXT,
    turn_id      INTEGER,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_log_created ON conversation_log(created_at);

CREATE TABLE IF NOT EXISTS kid_profile (
    id                  INTEGER PRIMARY KEY DEFAULT 1,
    nickname            TEXT,
    age                 INTEGER,
    total_turns         INTEGER DEFAULT 0,
    weakness_words      TEXT,
    dialogue_strategy   TEXT,
    in_refill_mode      INTEGER DEFAULT 0,
    last_new_word_turn  INTEGER DEFAULT 0,
    last_summary_turn   INTEGER DEFAULT 0,
    current_topic       TEXT,
    updated_at          TIMESTAMP,
    CHECK (id = 1)
);
"""


def _derive_status(
    current_status: Optional[str],
    mastery_score: int,
    is_target: bool = False,
) -> str:
    """Derive vocab_stats.status from inputs.

    'exposed':  passive exposure only, never been target, mastery < MASTERY_CAP
    'learning': has been target OR demoted from mastered, mastery < MASTERY_CAP
    'mastered': mastery_score >= MASTERY_CAP

    With CAP=2 there is no absorption buffer — mastery < 2 always means
    'not mastered', so a single wrong answer (2→1) demotes immediately.
    """
    if mastery_score >= MASTERY_CAP:
        return "mastered"
    if current_status == "mastered":
        # Below cap (mastery <= 1) and was mastered → demote to learning.
        # The conditional keeps the demotion threshold explicit in case CAP
        # is raised in the future; with CAP=2 the else branch is unreachable.
        return "learning" if mastery_score <= 1 else "mastered"
    if is_target:
        return "learning"
    if current_status is None:
        return "exposed"
    return current_status


class VocabRepo:
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def get_topics_for_word(self, word: str) -> List[str]:
        async with self._db.execute(
            "SELECT topic FROM ket_word_topics WHERE word = ? ORDER BY topic",
            (word,),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def get_ket_word(self, word: str) -> Optional[str]:
        """Case-insensitive lookup. Returns the canonical form stored in the
        vocab (e.g., 'i' → 'I'), or None if not found. The canonical form
        must be used by callers when tracking stats so mastery reconciles
        with target-word selection (which uses canonical form from the CSV).
        """
        async with self._db.execute(
            "SELECT word FROM ket_vocabulary WHERE word = ? COLLATE NOCASE LIMIT 1",
            (word,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    async def words_in_topic_without_stats(self, topic: str) -> List[str]:
        sql = (
            "SELECT v.word FROM ket_vocabulary v "
            "JOIN ket_word_topics t ON v.word = t.word "
            "LEFT JOIN vocab_stats s ON v.word = s.word "
            "WHERE t.topic = ? AND s.word IS NULL "
            "ORDER BY RANDOM() LIMIT 1"
        )
        async with self._db.execute(sql, (topic,)) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def unexposed_notopic_words(self) -> List[str]:
        sql = (
            "SELECT v.word FROM ket_vocabulary v "
            "WHERE NOT EXISTS (SELECT 1 FROM ket_word_topics t WHERE t.word = v.word) "
            "AND NOT EXISTS (SELECT 1 FROM vocab_stats s WHERE s.word = v.word) "
            "ORDER BY RANDOM() LIMIT 1"
        )
        async with self._db.execute(sql) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def topics_with_unmastered(self, exclude: Optional[str] = None) -> List[str]:
        sql = (
            "SELECT t.topic FROM ket_word_topics t "
            "LEFT JOIN vocab_stats s ON t.word = s.word "
            "WHERE (s.status IS NULL OR s.status != 'mastered')"
        )
        params = ()
        if exclude:
            sql += " AND t.topic != ?"
            params = (exclude,)
        sql += " GROUP BY t.topic ORDER BY RANDOM() LIMIT 1"
        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]


class StatsRepo:
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def get(self, word: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT word, exposed_count, correct_count, wrong_count, "
            "mastery_score, status, first_seen_at, last_seen_at "
            "FROM vocab_stats WHERE word = ?",
            (word,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "word": row[0],
            "exposed_count": row[1],
            "correct_count": row[2],
            "wrong_count": row[3],
            "mastery_score": row[4],
            "status": row[5],
            "first_seen_at": row[6],
            "last_seen_at": row[7],
        }

    async def apply_delta(
        self, word: str, delta: int, exposed: bool = False, is_target: bool = False
    ) -> dict:
        existing = await self.get(word)
        now = datetime.utcnow()
        if existing is None:
            score = min(MASTERY_CAP, max(0, delta))
            await self._db.execute(
                "INSERT INTO vocab_stats "
                "(word, exposed_count, correct_count, wrong_count, mastery_score, "
                "status, first_seen_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    word,
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
                "WHERE word=?",
                (
                    new_exposed,
                    new_correct,
                    new_wrong,
                    new_score,
                    _derive_status(existing["status"], new_score, is_target=is_target),
                    now,
                    word,
                ),
            )
        await self._db.commit()
        return await self.get(word)

    async def learning_count(self) -> int:
        async with self._db.execute(
            "SELECT COUNT(*) FROM vocab_stats WHERE status='learning'"
        ) as cur:
            row = await cur.fetchone()
        return row[0]

    async def oldest_learning_word(self) -> Optional[str]:
        # Two-tier: target words ('learning') take priority. Only when the
        # learning pool is dry do we promote the oldest scaffolding-only word
        # ('exposed') to target — this is the path that lets a previously-passive
        # word become an active target when the system has nothing else to practice.
        async with self._db.execute(
            "SELECT word FROM vocab_stats WHERE status='learning' "
            "ORDER BY last_seen_at ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            return row[0]
        async with self._db.execute(
            "SELECT word FROM vocab_stats WHERE status='exposed' "
            "ORDER BY last_seen_at ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def increment_exposed(self, word: str, is_target: bool = False) -> None:
        # delta=0 leaves mastery/correct/wrong unchanged; exposed=True
        # increments exposed_count; is_target threads through to status.
        await self.apply_delta(word, delta=0, exposed=True, is_target=is_target)


class ProfileRepo:
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def get(self) -> dict:
        async with self._db.execute(
            "SELECT nickname, age, total_turns, weakness_words, dialogue_strategy, "
            "in_refill_mode, last_new_word_turn, last_summary_turn, current_topic, updated_at "
            "FROM kid_profile WHERE id=1"
        ) as cur:
            row = await cur.fetchone()
        return {
            "nickname": row[0],
            "age": row[1],
            "total_turns": row[2],
            "weakness_words": json.loads(row[3]) if row[3] else [],
            "dialogue_strategy": row[4],
            "in_refill_mode": row[5],
            "last_new_word_turn": row[6],
            "last_summary_turn": row[7],
            "current_topic": row[8],
            "updated_at": row[9],
        }

    async def update(self, **fields) -> None:
        if not fields:
            return
        allowed = {
            "nickname", "age", "total_turns", "weakness_words",
            "dialogue_strategy", "in_refill_mode", "last_new_word_turn",
            "last_summary_turn", "current_topic",
        }
        set_parts = []
        values = []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in ("weakness_words",):
                v = json.dumps(v, ensure_ascii=False)
            set_parts.append(f"{k}=?")
            values.append(v)
        set_parts.append("updated_at=?")
        values.append(datetime.utcnow())
        await self._db.execute(
            f"UPDATE kid_profile SET {', '.join(set_parts)} WHERE id=1",
            values,
        )
        await self._db.commit()


class LogRepo:
    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def append(
        self,
        role: str,
        content: str,
        words_used: Optional[List[str]] = None,
        target_words: Optional[List[str]] = None,
        turn_id: Optional[int] = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO conversation_log (role, content, words_used, target_words, turn_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
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
            "FROM conversation_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
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
        """Return the most recent AI message IN THE CURRENT SESSION.

        A session boundary is marked by a row with role='system',
        content='session_start' (written by main.py on REPL startup).
        Only AI rows AFTER the most recent such marker are considered.
        If no marker exists yet (fresh DB), all AI rows are eligible —
        backward compat for databases populated before this feature.
        """
        async with self._db.execute(
            "SELECT content, words_used, target_words FROM conversation_log "
            "WHERE role='ai' AND id > COALESCE("
            "    (SELECT MAX(id) FROM conversation_log "
            "     WHERE role='system' AND content='session_start'),"
            "    0"
            ") ORDER BY id DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "content": row[0],
            "words_used": json.loads(row[1]) if row[1] else [],
            "target_words": json.loads(row[2]) if row[2] else [],
        }

    async def append_session_start(self) -> None:
        """Mark the start of a new REPL session. Subsequent calls to
        last_ai_message will ignore AI rows from before this marker,
        so a kid who exits mid-sentence won't be shown the explanation
        of that sentence on restart.
        """
        await self.append(role="system", content="session_start")


class Repos:
    def __init__(self, db: aiosqlite.Connection):
        self.vocab = VocabRepo(db)
        self.stats = StatsRepo(db)
        self.profile = ProfileRepo(db)
        self.log = LogRepo(db)
        self._db = db

    async def close(self):
        await self._db.close()


_DEFAULT_CSV = join(dirname(__file__), "..", "..", "..", "data", "KET_vocabulary.csv")


async def init_db(db_path: str, csv_path: Optional[str] = None) -> Repos:
    db = await aiosqlite.connect(db_path)
    db.row_factory = sqlite3.Row
    await db.executescript(_SCHEMA)
    await db.execute(
        "INSERT OR IGNORE INTO kid_profile (id, total_turns) VALUES (1, 0)"
    )
    await db.commit()

    if csv_path:
        await _import_csv(db, csv_path)

    return Repos(db)


async def _import_csv(db: aiosqlite.Connection, csv_path: str) -> None:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            word = (row.get("word") or "").strip()
            pos = (row.get("part_of_speech") or "").strip()
            topic_raw = (row.get("topic") or "").strip()
            if not word:
                continue
            await db.execute(
                "INSERT OR IGNORE INTO ket_vocabulary (word, pos, is_seed) VALUES (?, ?, 0)",
                (word, pos),
            )
            if topic_raw:
                for t in topic_raw.split(";"):
                    t = t.strip()
                    if t:
                        await db.execute(
                            "INSERT OR IGNORE INTO ket_word_topics (word, topic) VALUES (?, ?)",
                            (word, t),
                        )
            count += 1
        await db.commit()
        logger.info(f"Imported {count} words from {csv_path}")
