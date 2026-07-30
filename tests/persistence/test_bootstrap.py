# tests/persistence/test_bootstrap.py
import tempfile

import pytest

from src.persistence.bootstrap import init_db


@pytest.mark.asyncio
async def test_init_db_creates_tables(temp_db_path):
    db = await init_db(temp_db_path, csv_path=None)
    try:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cur:
            rows = await cur.fetchall()
        table_names = {r[0] for r in rows}
        assert "ket_vocabulary" in table_names
        assert "ket_vocab_topics" in table_names
        assert "vocab_stats" in table_names
        assert "conversation_log" in table_names
        assert "kid_profile" in table_names
        assert "users" in table_names
        assert "recent_sentences" in table_names
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_import_csv_creates_one_row_per_context(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic,context\n"
        "design,n,,planning\n"
        "design,n,,process\n"
        "design,n,Entertainment and Media,drawing\n"
        "design,v,,\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    try:
        async with db.execute(
            "SELECT context, pos FROM ket_vocabulary WHERE word='design' ORDER BY context"
        ) as cur:
            rows = await cur.fetchall()
        assert [tuple(r) for r in rows] == [
            ("", "v"),
            ("drawing", "n"),
            ("planning", "n"),
            ("process", "n"),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_import_csv_handles_bom(temp_db_path):
    csv_text = (
        "﻿word,part_of_speech,topic,context\n"
        "cat,n,Animals,\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    try:
        async with db.execute(
            "SELECT word FROM ket_vocabulary WHERE word='cat' LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "BOM must not prevent 'cat' from being read"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_import_csv_links_topic_to_specific_context(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic,context\n"
        "smart,adj,,stylish\n"
        'smart,adj,"Education",clever\n'
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    try:
        clever_topics: list[str]
        stylish_topics: list[str]
        async with db.execute(
            "SELECT topic FROM ket_vocab_topics WHERE word=? AND context=? "
            "ORDER BY topic",
            ("smart", "clever"),
        ) as cur:
            rows = await cur.fetchall()
        clever_topics = [r[0] for r in rows]
        async with db.execute(
            "SELECT topic FROM ket_vocab_topics WHERE word=? AND context=? "
            "ORDER BY topic",
            ("smart", "stylish"),
        ) as cur:
            rows = await cur.fetchall()
        stylish_topics = [r[0] for r in rows]
        assert "Education" in clever_topics
        assert stylish_topics == []
    finally:
        await db.close()
