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


@pytest.mark.asyncio
async def test_migration_adds_user_id_to_conversation_log(temp_db_path):
    """Legacy DB: conversation_log without user_id column. Migration must add it
    with default 'default' and preserve existing rows."""
    db = await aiosqlite.connect(temp_db_path)
    # Legacy conversation_log schema: no user_id column.
    await db.execute(
        "CREATE TABLE conversation_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    await db.execute(
        "INSERT INTO conversation_log (role, content) VALUES ('user', 'hello')"
    )
    await db.execute(
        "INSERT INTO conversation_log (role, content) VALUES ('ai', 'hi kid')"
    )
    await db.commit()
    await migrate_old_schema_if_needed(db)

    # Schema assertion: user_id column now present.
    async with db.execute("PRAGMA table_info(conversation_log)") as cur:
        cols = [r[1] for r in await cur.fetchall()]
    assert "user_id" in cols

    # Data preservation: both legacy rows survive and get the default user_id.
    async with db.execute(
        "SELECT user_id, role, content FROM conversation_log ORDER BY id"
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 2, f"both legacy rows must survive, got {len(rows)}"
    assert all(r[0] == "default" for r in rows), (
        f"legacy rows must backfill to user_id='default', got {[r[0] for r in rows]}"
    )
    assert (rows[0][1], rows[0][2]) == ("user", "hello")
    assert (rows[1][1], rows[1][2]) == ("ai", "hi kid")
    await db.close()


@pytest.mark.asyncio
async def test_migration_rebuilds_legacy_kid_profile(temp_db_path):
    """Legacy DB: kid_profile has `id` column (single-user) and no user_id.
    Migration must DROP + rebuild kid_profile with user_id PRIMARY KEY, create
    the users table, and copy legacy data into both."""
    db = await aiosqlite.connect(temp_db_path)
    # Legacy kid_profile schema: integer id PK, no user_id, same 8 data columns
    # the migration reads (total_turns, weakness_words, dialogue_strategy,
    # in_refill_mode, last_new_word_turn, last_summary_turn, current_topic,
    # updated_at) plus nickname/age which the migration reads off the id=1 row.
    await db.execute(
        "CREATE TABLE kid_profile ("
        "id INTEGER PRIMARY KEY, "
        "nickname TEXT, age INTEGER, "
        "total_turns INTEGER, weakness_words TEXT, dialogue_strategy TEXT, "
        "in_refill_mode INTEGER, last_new_word_turn INTEGER, "
        "last_summary_turn INTEGER, current_topic TEXT, updated_at TIMESTAMP)"
    )
    await db.execute(
        "INSERT INTO kid_profile (id, nickname, age, total_turns, weakness_words, "
        "dialogue_strategy, in_refill_mode, last_new_word_turn, last_summary_turn, "
        "current_topic, updated_at) VALUES "
        "(1, '小明', 9, 7, '[\"cat\"]', 'be patient', 1, 5, 3, 'Animals', '2026-01-01')"
    )
    await db.commit()
    await migrate_old_schema_if_needed(db)

    # Schema assertion: kid_profile now keyed on user_id, no `id` column.
    async with db.execute("PRAGMA table_info(kid_profile)") as cur:
        prof_cols = [r[1] for r in await cur.fetchall()]
    assert "user_id" in prof_cols
    assert "id" not in prof_cols, "legacy id column must be gone after rebuild"

    # Schema assertion: users table created with the multi-tenant shape.
    async with db.execute("PRAGMA table_info(users)") as cur:
        user_cols = [r[1] for r in await cur.fetchall()]
    assert user_cols, "users table must exist after migration"
    assert "id" in user_cols and "nickname" in user_cols and "age" in user_cols

    # Data preservation: legacy nickname/age landed in users under 'default'.
    async with db.execute(
        "SELECT id, nickname, age FROM users WHERE id='default'"
    ) as cur:
        user_row = await cur.fetchone()
    assert user_row is not None, "legacy profile must seed users('default')"
    assert (user_row[1], user_row[2]) == ("小明", 9), (
        f"nickname/age must migrate to users row, got {user_row}"
    )

    # Data preservation: legacy turn-state landed in kid_profile under 'default'.
    async with db.execute(
        "SELECT total_turns, weakness_words, dialogue_strategy, current_topic "
        "FROM kid_profile WHERE user_id='default'"
    ) as cur:
        prof_row = await cur.fetchone()
    assert prof_row is not None, "legacy turn-state must migrate to kid_profile('default')"
    assert prof_row[0] == 7
    assert prof_row[1] == '["cat"]'
    assert prof_row[2] == "be patient"
    assert prof_row[3] == "Animals"
    await db.close()
