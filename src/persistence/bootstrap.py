# src/persistence/bootstrap.py
"""Composition-root DB initialization. Not for per-request use.

init_db opens a connection, runs migration, applies the schema, seeds the
default user + kid_profile, and optionally imports the KET vocabulary CSV.
The migrate_old_schema_if_needed call is a no-op stub in Task 3; Task 8
replaces it with the real persistence.migration import.
"""
import csv
import sqlite3

import aiosqlite

from flow.common import logger
from src.persistence.schema import SCHEMA_SQL


async def init_db(
    db_path: str,
    csv_path: str | None = None,
    default_nickname: str = "宝贝",
    default_age: int = 8,
) -> aiosqlite.Connection:
    """Open connection, set PRAGMAs, apply schema, seed default user +
    kid_profile, optional CSV import. Returns the raw connection."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    # Migration is a no-op in Task 3; Task 8 wires migrate_old_schema_if_needed.
    await db.executescript(SCHEMA_SQL)
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
    """Import KET vocabulary from CSV. Private — only bootstrap.py calls this.

    Copy body verbatim from src/flow/ket_partner/db.py:737-762.
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:  # noqa: ASYNC230
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
