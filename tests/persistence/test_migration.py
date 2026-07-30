# tests/persistence/test_migration.py
import aiosqlite
import pytest

from src.persistence.migration import migrate_old_schema_if_needed
from src.persistence.schema import SCHEMA_SQL


@pytest.mark.asyncio
async def test_migration_adds_user_id_to_vocab_stats(temp_db_path):
    """Legacy DB: vocab_stats without user_id column. Migration must add it."""
    db = await aiosqlite.connect(temp_db_path)
    await db.execute(
        "CREATE TABLE vocab_stats (word TEXT, context TEXT, exposed_count INTEGER)"
    )
    await db.execute(
        "INSERT INTO vocab_stats (word, context, exposed_count) VALUES ('cat', '', 3)"
    )
    await db.commit()
    await migrate_old_schema_if_needed(db)

    async with db.execute("PRAGMA table_info(vocab_stats)") as cur:
        cols = [r[1] for r in await cur.fetchall()]
    assert "user_id" in cols

    async with db.execute("SELECT user_id FROM vocab_stats WHERE word='cat'") as cur:
        row = await cur.fetchone()
    assert row is not None and row[0] == "default"
    await db.close()


@pytest.mark.asyncio
async def test_migration_noop_on_modern_schema(temp_db_path):
    """Modern DB (already has user_id columns) → migration is a no-op."""
    db = await aiosqlite.connect(temp_db_path)
    await db.executescript(SCHEMA_SQL)
    await db.commit()
    # Should not raise
    await migrate_old_schema_if_needed(db)
    await db.close()
