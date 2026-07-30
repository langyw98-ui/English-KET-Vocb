"""Schema DDL for the KET partner persistence layer.

Single source of truth for table + index definitions; consumed by
persistence.bootstrap.init_db via executescript.
"""

SCHEMA_SQL: str = """
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
