# tests/persistence/test_repos.py
import tempfile

import pytest

from src.persistence.bootstrap import init_db
from src.persistence.models import WordRef
from src.persistence.repos import VocabRepo


async def _init_vocab_repo(db_path: str, csv_path: str | None = None) -> VocabRepo:
    """Open DB via bootstrap.init_db, wrap a VocabRepo around it.

    Each test owns the connection and must close it in a finally block,
    OR rely on the temp_db_path fixture's cleanup (file-based DB). Tests
    that need stats/log/recent access should NOT use this helper — they
    belong to later tasks.
    """
    db = await init_db(db_path, csv_path=csv_path)
    return VocabRepo(db, "default")


@pytest.mark.asyncio
class TestVocabRepo:
    async def test_get_ket_word_returns_wordref_with_context(self, temp_db_path):
        """Spec §4.1: get_ket_word returns a WordRef carrying both word and
        context so callers don't have to thread context separately."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "smart,adj,,clever\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        wr = await repo.get_ket_word("smart", context="clever")
        assert wr == WordRef(word="smart", context="clever")

    async def test_get_ket_word_returns_none_when_context_mismatch(self, temp_db_path):
        """Spec §5.1: precise lookup. smart has only (smart, clever); asking
        for (smart, '') returns None — this is the property the validator
        MUST avoid by using get_ket_word_any_context."""
        csv_text = "word,part_of_speech,topic,context\nsmart,adj,,clever\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        assert await repo.get_ket_word("smart", context="") is None

    async def test_get_ket_word_any_context_prefers_default(self, temp_db_path):
        """Spec §5.1: when (word, '') exists, return it; otherwise return the
        first by context ASC. Preferring default keeps the any-context lookup
        stable across schema growth."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "design,v,,\n"            # default sense
            "design,n,,planning\n"
            "design,n,,drawing\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        wr = await repo.get_ket_word_any_context("design")
        assert wr == WordRef(word="design", context="")

    async def test_get_ket_word_any_context_returns_first_when_no_default(self, temp_db_path):
        """Spec §5.1: 7 orphan-skip words have no (word, '') row. The
        any-context lookup must still recognize them as KET — return the
        lexicographically first context."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "smart,adj,,stylish\n"
            "smart,adj,,clever\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        wr = await repo.get_ket_word_any_context("smart")
        assert wr == WordRef(word="smart", context="clever")

    async def test_words_in_topic_without_stats_returns_wordref(self, temp_db_path):
        """Spec §6: vocab_selector transparently threads WordRef through; the
        underlying query must return (word, context) tuples.

        Uses a single-candidate CSV because the SQL is ORDER BY RANDOM() LIMIT 1
        (production indexes [0]) -- deterministic assertion needs exactly one
        candidate."""
        csv_text = (
            "word,part_of_speech,topic,context\n"
            "cat,n,Animals,\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(csv_text)
            csv_path = f.name
        repo = await _init_vocab_repo(temp_db_path, csv_path=csv_path)
        candidates = await repo.words_in_topic_without_stats("Animals")
        assert candidates == [WordRef(word="cat", context="")]
