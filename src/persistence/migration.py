# src/persistence/migration.py
"""Legacy single-user schema → multi-tenant migration.

Detected via missing user_id columns on vocab_stats / conversation_log,
or kid_profile having `id` column instead of `user_id`.
"""
import aiosqlite

from flow.common import logger  # logger live in flow.common; OK to import


async def migrate_old_schema_if_needed(db: aiosqlite.Connection) -> None:
    """自动检测并升级旧版本 CLI 数据库 (补齐 user_id 列并转换 kid_profile 表)"""
    # 检查 vocab_stats 是否存在但缺失 user_id
    async with db.execute("PRAGMA table_info(vocab_stats)") as cur:
        cols_stats = [r[1] for r in await cur.fetchall()]
    if cols_stats and "user_id" not in cols_stats:
        logger.info("Migrating legacy vocab_stats table to include user_id...")
        await db.execute("ALTER TABLE vocab_stats ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")

    # 检查 conversation_log 是否存在但缺失 user_id
    async with db.execute("PRAGMA table_info(conversation_log)") as cur:
        cols_log = [r[1] for r in await cur.fetchall()]
    if cols_log and "user_id" not in cols_log:
        logger.info("Migrating legacy conversation_log table to include user_id...")
        await db.execute("ALTER TABLE conversation_log ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")

    # 检查 kid_profile 是否为旧结构 (有 id 列但缺失 user_id 列)
    async with db.execute("PRAGMA table_info(kid_profile)") as cur:
        cols_prof = [r[1] for r in await cur.fetchall()]
    if cols_prof and "user_id" not in cols_prof:
        logger.info("Migrating legacy kid_profile table to multi-tenant schema...")
        async with db.execute(
            "SELECT nickname, age, total_turns, weakness_words, dialogue_strategy, "
            "in_refill_mode, last_new_word_turn, last_summary_turn, current_topic, updated_at "
            "FROM kid_profile WHERE id=1"
        ) as cur:
            row = await cur.fetchone()

        nickname = (row[0] if row else None) or "宝贝"
        age = (row[1] if row else None) or 8

        await db.execute(
            "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, nickname TEXT NOT NULL, age INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        await db.execute(
            "INSERT OR IGNORE INTO users (id, nickname, age) VALUES ('default', ?, ?)",
            (nickname, age),
        )
        await db.execute("DROP TABLE kid_profile")
        await db.execute(
            "CREATE TABLE IF NOT EXISTS kid_profile ("
            "user_id TEXT PRIMARY KEY, "
            "total_turns INTEGER DEFAULT 0, "
            "weakness_words TEXT, "
            "dialogue_strategy TEXT, "
            "in_refill_mode INTEGER DEFAULT 0, "
            "last_new_word_turn INTEGER DEFAULT 0, "
            "last_summary_turn INTEGER DEFAULT 0, "
            "current_topic TEXT, "
            "updated_at TIMESTAMP)"
        )
        if row:
            await db.execute(
                "INSERT OR IGNORE INTO kid_profile "
                "(user_id, total_turns, weakness_words, dialogue_strategy, in_refill_mode, last_new_word_turn, last_summary_turn, current_topic, updated_at) "
                "VALUES ('default', ?, ?, ?, ?, ?, ?, ?, ?)",
                (row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]),
            )
    await db.commit()
