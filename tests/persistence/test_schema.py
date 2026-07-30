# tests/persistence/test_schema.py
from src.persistence.schema import SCHEMA_SQL


def test_schema_sql_contains_all_tables():
    for table in (
        "ket_vocabulary", "ket_vocab_topics", "vocab_stats",
        "conversation_log", "kid_profile", "users", "recent_sentences",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA_SQL, (
            f"missing DDL for {table}"
        )


def test_schema_sql_has_indexes():
    for idx in (
        "idx_vocab_seed", "idx_vocab_topics_topic", "idx_vocab_topics_lookup",
        "idx_stats_user_status", "idx_log_user_id", "idx_recent_user_created",
    ):
        assert idx in SCHEMA_SQL, f"missing index {idx}"
