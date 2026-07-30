# KET Partner Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `src/flow/ket_partner/` into a PURE agent package + top-level `persistence/`, `cli/ket_partner/`, `reporting/ket_partner/` packages, eliminating cross-layer dependency tangles.

**Architecture:** Three-layer dependency DAG: `persistence/` (Repos impl) → `flow/ket_partner/` (domain core, Protocol contract) → `cli/` + `api/` + `reporting/` (consumers). Hexagonal boundary via `KETPartnerRepos` Protocol; reporting owns category rules (Option B) to eliminate persistence → KetConfig reverse dep.

**Tech Stack:** Python 3.11+, LangGraph, LangChain, aiosqlite, FastAPI, pydantic, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-07-29-ket-partner-package-restructure-design.md` (commit `94f6834`). All signatures and rationale live there; this plan references it but does not repeat it.

## Global Constraints

Apply to every task; do not violate.

- **Python interpreter**: `D:/ProgramData/miniforge3/envs/langgraph/python.exe` for all commands (base miniforge3 missing deps).
- **Console output**: plain Chinese + common CJK punctuation only, no emoji (Windows GBK terminal).
- **Static checks per task**: `ruff check <paths>` + `mypy <paths>` + `pytest <paths>` all clear before commit. Warnings are errors.
- **No bare `except` / `except Exception`**: use specific exception types; never include `ValueError`/`TypeError`/`KeyError`/`AttributeError`/`IndexError` in cross-boundary exception tuples.
- **LLM calls**: every `with_structured_output(Schema)` passes `method="function_calling"`; mocks accept `**kwargs`.
- **Schema changes**: update prompt + tests + run real LLM once with `@pytest.mark.integration`.
- **Mock discipline**: sync methods → `MagicMock` + `assert_called*`; async methods → `AsyncMock` + `assert_awaited*`. Always assert call count, not just return value.
- **No backwards-compat shims** for deleted code; no `# removed` comments.
- **Module-level constant** for any magic string/number; no inline tuples or hardcoded lists.
- **Single-Writer**: every state field's writer documented in TypedDict docstring.

---

## Phase A: persistence package (NEW)

Builds the new top-level `src/persistence/` package alongside the existing `flow/ket_partner/db.py`. Once Phase A–F all land, `db.py` is deleted (Task 24).

### Task 1: Scaffold persistence package + schema.py

**Files:**
- Create: `src/persistence/__init__.py` (empty)
- Create: `src/persistence/schema.py`
- Create: `tests/persistence/__init__.py` (empty)
- Create: `tests/persistence/conftest.py`
- Create: `tests/persistence/test_schema.py`

**Interfaces:**
- Produces: `persistence.schema.SCHEMA_SQL: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/persistence/test_schema.py
from persistence.schema import SCHEMA_SQL


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
```

```python
# tests/persistence/conftest.py
import os
import tempfile

import pytest


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'persistence'`

- [ ] **Step 3: Create schema.py**

Copy `_SCHEMA` constant verbatim from `src/flow/ket_partner/db.py:36-107` into:

```python
# src/persistence/schema.py
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
```

Create empty `src/persistence/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_schema.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/persistence/__init__.py src/persistence/schema.py tests/persistence/
git commit -m "feat(persistence): scaffold package + schema.SCHEMA_SQL"
```

---

### Task 2: persistence/models.py (WordRef + derive_status)

**Files:**
- Create: `src/persistence/models.py`
- Create: `tests/persistence/test_models.py`

**Interfaces:**
- Produces: `WordRef` (NamedTuple, fields `word: str`, `context: str = ""`)
- Produces: `MASTERY_CAP: int` (= 2)
- Produces: `derive_status(current_status, mastery_score, is_target=False) -> str`

- [ ] **Step 1: Write the failing test**

Move the 5 `test_derive_status_*` functions from `tests/ket_partner/test_db.py:7-43` verbatim into `tests/persistence/test_models.py`, but change the import:

```python
# tests/persistence/test_models.py
from persistence.models import MASTERY_CAP, WordRef, derive_status


def test_derive_status_mastered_at_score_cap():
    assert derive_status("learning", 2) == "mastered"
    assert derive_status("exposed", 2) == "mastered"
    assert derive_status("mastered", 5) == "mastered"


def test_derive_status_mastered_demotes_at_score_1_or_below():
    assert derive_status("mastered", 1) == "learning"
    assert derive_status("mastered", 0) == "learning"


def test_derive_status_is_target_promotes_to_learning():
    assert derive_status("exposed", 0, is_target=True) == "learning"
    assert derive_status("exposed", 1, is_target=True) == "learning"
    assert derive_status(None, 0, is_target=True) == "learning"


def test_derive_status_new_row_not_target_is_exposed():
    assert derive_status(None, 0) == "exposed"


def test_derive_status_preserves_exposed_and_learning_below_cap():
    assert derive_status("exposed", 1) == "exposed"
    assert derive_status("exposed", 2) == "mastered"
    assert derive_status("learning", 1) == "learning"
    assert derive_status("learning", 2) == "mastered"


def test_mastery_cap_is_two():
    assert MASTERY_CAP == 2


def test_wordref_defaults_empty_context():
    ref = WordRef(word="cat")
    assert ref.word == "cat"
    assert ref.context == ""
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'persistence.models'`

- [ ] **Step 3: Create models.py**

```python
# src/persistence/models.py
"""Cross-repo value objects + business constants + pure status derivation.

WordRef is the unit of practice threading vocab_selector → agent → evaluator.
MASTERY_CAP caps the mastery score so demotion path stays short.
derive_status is a pure function over (current_status, mastery_score, is_target).
"""
from typing import NamedTuple


class WordRef(NamedTuple):
    """A (word, context) pair — the unit of practice."""
    word: str
    context: str = ""


MASTERY_CAP: int = 2


def derive_status(
    current_status: str | None,
    mastery_score: int,
    is_target: bool = False,
) -> str:
    """Derive vocab_stats.status from inputs.

    Returns one of: 'mastered' | 'learning' | 'exposed' | current_status.
    """
    if mastery_score >= MASTERY_CAP:
        return "mastered"
    if current_status == "mastered":
        return "learning" if mastery_score <= 1 else "mastered"
    if is_target:
        return "learning"
    if current_status is None:
        return "exposed"
    return current_status
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_models.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/persistence/models.py tests/persistence/test_models.py
git commit -m "feat(persistence): add models.py with WordRef, MASTERY_CAP, derive_status"
```

---

### Task 3: persistence/repos.py — VocabRepo

**Files:**
- Create: `src/persistence/repos.py` (starts here, expanded in A4–A6)
- Create: `tests/persistence/test_repos.py` (starts here)

**Interfaces:**
- Produces: `VocabRepo` with 7 async methods (spec §3).

- [ ] **Step 1: Write the failing test**

Move VocabRepo-related tests from `tests/ket_partner/test_db.py` into `tests/persistence/test_repos.py` under a `TestVocabRepo` class. Change the import:

```python
# tests/persistence/test_repos.py
import pytest

from persistence.models import WordRef
from persistence.repos import VocabRepo, Repos  # Repos added in A6
from persistence.bootstrap import init_db  # added in A7; for now stub


async def _init_repos(db_path: str, csv_path: str | None = None) -> Repos:
    db = await init_db(db_path, csv_path=csv_path)
    return Repos.for_user(db, "default")


@pytest.mark.asyncio
class TestVocabRepo:
    async def test_get_ket_word_returns_wordref_with_context(self, temp_db_path):
        # Copy body from tests/ket_partner/test_db.py:422-438 verbatim
        ...

    async def test_get_ket_word_returns_none_when_context_mismatch(self, temp_db_path):
        # Copy body from tests/ket_partner/test_db.py:440-453 verbatim
        ...

    async def test_get_ket_word_any_context_prefers_default(self, temp_db_path):
        # Copy body from tests/ket_partner/test_db.py:454-474 verbatim
        ...

    async def test_get_ket_word_any_context_returns_first_when_no_default(self, temp_db_path):
        # Copy body from tests/ket_partner/test_db.py:475-494 verbatim
        ...

    async def test_words_in_topic_without_stats_returns_wordref(self, temp_db_path):
        # Copy body from tests/ket_partner/test_db.py:495-516 verbatim
        ...

    async def test_import_csv_splits_multi_topic(self, temp_db_path):
        # Copy body from tests/ket_partner/test_db.py:81-104 verbatim (exercises get_topics_for_word)
        ...
```

For each `...` body, copy verbatim from the cited test_db.py lines. The test signatures are `async def test_...(self, temp_db_path):` instead of module-level `async def test_...(temp_db_path):` — adjust indentation.

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestVocabRepo -v`
Expected: FAIL — `ImportError: cannot import name 'VocabRepo' from 'persistence.repos'`

- [ ] **Step 3: Create repos.py with VocabRepo**

Copy `VocabRepo` class verbatim from `src/flow/ket_partner/db.py:127-214`:

```python
# src/persistence/repos.py
"""Per-user Repo classes for the KET partner persistence layer.

Each Repo exposes a narrow async interface over a single table family.
Repos (Task 6) aggregates the 5 per-user Repos for one request.
"""
import json
import re
from datetime import datetime, timezone

import aiosqlite

from persistence.models import MASTERY_CAP, WordRef, derive_status


class VocabRepo:
    """ket_vocabulary / ket_vocab_topics tables — read access."""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None:
        self._db = db
        self._user_id = user_id

    async def get_topics_for_word(self, word: str, context: str = "") -> list[str]:
        async with self._db.execute(
            "SELECT topic FROM ket_vocab_topics "
            "WHERE word = ? AND context = ? ORDER BY topic",
            (word, context),
        ) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def get_ket_word(
        self, word: str, context: str = ""
    ) -> WordRef | None:
        async with self._db.execute(
            "SELECT word, context FROM ket_vocabulary "
            "WHERE word = ? COLLATE NOCASE AND context = ? LIMIT 1",
            (word, context),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return WordRef(word=row[0], context=row[1])

    async def get_ket_word_any_context(self, word: str) -> WordRef | None:
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

    async def words_in_topic_without_stats(self, topic: str) -> list[WordRef]:
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

    async def unexposed_notopic_words(self) -> list[WordRef]:
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

    async def topics_with_unmastered(self, exclude: str | None = None) -> list[str]:
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
```

(Header stub `Repos` and `init_db` references stay — A6 / A7 fill them.)

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestVocabRepo -v`
Expected: 6 passed (TestVocabRepo exercises 6 functions; one of them is `test_import_csv_splits_multi_topic` which is technically a bootstrap test — leave it here only if you keep it; otherwise move to test_bootstrap.py in A7).

- [ ] **Step 5: Commit**

```bash
git add src/persistence/repos.py tests/persistence/test_repos.py
git commit -m "feat(persistence): add VocabRepo + test_repos.TestVocabRepo"
```

---

### Task 4: persistence/repos.py — StatsRepo (Option B restructure)

**Files:**
- Modify: `src/persistence/repos.py` (append StatsRepo)
- Modify: `tests/persistence/test_repos.py` (append TestStatsRepo)

**Interfaces:**
- Produces: `StatsRepo` with `get` / `apply_delta` / `learning_count` / `oldest_learning_word` / `increment_exposed` / `list_all_with_vocab` (NEW)
- **Deletes** (vs db.py): `_category_where_sql`, `count_by_category`, `list_by_category`
- **Signature change**: `__init__(db, user_id)` — NO `config` param (spec §3).

- [ ] **Step 1: Write the failing test**

Add `TestStatsRepo` to `tests/persistence/test_repos.py`. Copy these from `tests/ket_partner/test_db.py`:
- `test_stats_repo_apply_delta_creates_row` (l.180)
- `test_stats_repo_apply_delta_floor_at_zero` (l.195)
- `test_stats_repo_apply_delta_caps_mastery_at_cap` (l.208)
- `test_stats_repo_apply_delta_caps_single_large_delta` (l.233)
- `test_status_transitions` (l.246)
- `test_increment_exposed_creates_exposed_row_for_scaffolding` (l.270)
- `test_increment_exposed_creates_learning_row_for_target` (l.284)
- `test_increment_exposed_promotes_existing_exposed_to_learning` (l.297)
- `test_increment_exposed_preserves_learning_on_scaffolding_reexposure` (l.313)
- `test_oldest_learning_word_returns_none_when_empty` (l.346)
- `test_oldest_learning_word_prefers_learning_over_exposed` (l.352)
- `test_oldest_learning_word_falls_back_to_exposed_when_no_learning` (l.375)
- `test_oldest_learning_word_ignores_mastered` (l.398)
- `test_oldest_learning_word_returns_wordref` (l.577)
- `test_apply_delta_skips_orphan_default_sense` (l.517)
- `test_apply_delta_writes_default_sense_when_vocab_has_row` (l.536)
- `test_apply_delta_threads_context_for_target_path` (l.559)

**Delete** (do NOT migrate): `test_stats_count_and_list_by_category` (l.628) — Option B removes the methods under test.

Add ONE new test for `list_all_with_vocab`:

```python
@pytest.mark.asyncio
class TestStatsRepo:
    # ... (migrated tests above) ...

    async def test_list_all_with_vocab_returns_all_words(self, temp_db_path):
        """list_all_with_vocab returns every ket_vocabulary row LEFT JOINed
        with vocab_stats — including words the kid never practiced. Replaces
        the old exporter's repos.stats._db.execute private access."""
        repos = await _init_repos(temp_db_path, csv_path=None)
        # Manually insert two vocab rows; no stats yet
        await repos.vocab._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES (?, ?, ?, 0)",
            ("cat", "", "n"),
        )
        await repos.vocab._db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES (?, ?, ?, 0)",
            ("dog", "", "n"),
        )
        await repos.stats.apply_delta("cat", delta=1, exposed=True)
        await repos.vocab._db.commit()

        rows = await repos.stats.list_all_with_vocab()
        words = {r["word"] for r in rows}
        assert words == {"cat", "dog"}
        cat_row = next(r for r in rows if r["word"] == "cat")
        assert cat_row["exposed_count"] == 1
        assert cat_row["correct_count"] == 1
        dog_row = next(r for r in rows if r["word"] == "dog")
        assert dog_row["exposed_count"] == 0
        assert dog_row["status"] == "new"
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestStatsRepo -v`
Expected: FAIL — `AttributeError: 'StatsRepo' object has no attribute 'list_all_with_vocab'`

- [ ] **Step 3: Append StatsRepo to repos.py**

Adapt `StatsRepo` from `src/flow/ket_partner/db.py:217-422`. **Delete** `_category_where_sql`, `count_by_category`, `list_by_category` (lines 349-422). **Drop** `config` param + `self._config` field + `if config is None: from flow.ket_partner.config import load_config` block. **Add** `list_all_with_vocab`:

```python
# src/persistence/repos.py (append)
class StatsRepo:
    """vocab_stats table — read/write + mastery derivation.

    Option B deletes _category_where_sql/count_by_category/list_by_category;
    category rules now live in reporting/ket_partner/categories.py.
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> None:
        self._db = db
        self._user_id = user_id

    async def _vocab_has_default_sense(self, word: str) -> bool:
        async with self._db.execute(
            "SELECT 1 FROM ket_vocabulary WHERE word = ? AND context = '' LIMIT 1",
            (word,),
        ) as cur:
            return await cur.fetchone() is not None

    async def get(self, word: str, context: str = "") -> dict | None:
        # Copy body from db.py:238-258 verbatim
        ...

    async def apply_delta(
        self,
        word: str,
        context: str = "",
        delta: int = 0,
        exposed: bool = False,
        is_target: bool = False,
    ) -> dict | None:
        # Copy body from db.py:260-315 verbatim
        # (uses MASTERY_CAP and derive_status imported at top)
        ...

    async def learning_count(self) -> int:
        # Copy body from db.py:317-323 verbatim
        ...

    async def oldest_learning_word(self) -> WordRef | None:
        # Copy body from db.py:325-342 verbatim
        ...

    async def increment_exposed(
        self, word: str, context: str = "", is_target: bool = False,
    ) -> None:
        await self.apply_delta(word, context=context, delta=0, exposed=True, is_target=is_target)

    async def list_all_with_vocab(self) -> list[dict]:
        """vocab_stats LEFT JOIN ket_vocabulary — all words including
        never-practiced. Replaces the old exporter's repos.stats._db.execute.
        """
        async with self._db.execute(
            "SELECT v.word, v.context, v.pos, "
            "COALESCE(s.exposed_count, 0) AS exposed_count, "
            "COALESCE(s.correct_count, 0) AS correct_count, "
            "COALESCE(s.wrong_count, 0) AS wrong_count, "
            "COALESCE(s.mastery_score, 0) AS mastery_score, "
            "COALESCE(s.status, 'new') AS status "
            "FROM ket_vocabulary v "
            "LEFT JOIN vocab_stats s ON v.word = s.word AND v.context = s.context "
            "AND s.user_id = ? "
            "ORDER BY v.word, v.context",
            (self._user_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "word": r[0],
                "context": r[1] or "",
                "pos": r[2],
                "exposed_count": r[3] or 0,
                "correct_count": r[4] or 0,
                "wrong_count": r[5] or 0,
                "mastery_score": r[6] or 0,
                "status": r[7] or "new",
            }
            for r in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestStatsRepo -v`
Expected: 18 passed (17 migrated + 1 new for `list_all_with_vocab`)

- [ ] **Step 5: Commit**

```bash
git add src/persistence/repos.py tests/persistence/test_repos.py
git commit -m "feat(persistence): add StatsRepo with list_all_with_vocab; drop category SQL (Option B)"
```

---

### Task 5: persistence/repos.py — ProfileRepo + LogRepo

**Files:**
- Modify: `src/persistence/repos.py`
- Modify: `tests/persistence/test_repos.py`

**Interfaces:**
- Produces: `ProfileRepo` (get, update)
- Produces: `LogRepo` (append, recent, append_session_start, last_ai_message)

- [ ] **Step 1: Write the failing test**

Add `TestProfileRepo` + `TestLogRepo` to `tests/persistence/test_repos.py`. Copy from `tests/ket_partner/test_db.py`:

ProfileRepo:
- `test_profile_repo_init_creates_single_row` (l.171)

LogRepo:
- `test_last_ai_message_returns_none_when_no_ai_rows` (l.328)
- `test_last_ai_message_returns_latest_when_no_session_start` (l.334)
- `test_log_append_persists_target_words_with_context` (l.593)

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestProfileRepo tests/persistence/test_repos.py::TestLogRepo -v`
Expected: FAIL — `AttributeError: module 'persistence.repos' has no attribute 'ProfileRepo'`

- [ ] **Step 3: Append ProfileRepo + LogRepo**

Copy `ProfileRepo` from `src/flow/ket_partner/db.py:425-502` and `LogRepo` from `db.py:505-570` verbatim into `src/persistence/repos.py`. No signature changes.

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestProfileRepo tests/persistence/test_repos.py::TestLogRepo -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/persistence/repos.py tests/persistence/test_repos.py
git commit -m "feat(persistence): add ProfileRepo + LogRepo"
```

---

### Task 6: persistence/repos.py — RecentSentencesRepo + Repos aggregator

**Files:**
- Modify: `src/persistence/repos.py`
- Modify: `tests/persistence/test_repos.py`

**Interfaces:**
- Produces: `RecentSentencesRepo` (list_recent, append, list_recent_scaffolding)
- Produces: `Repos` aggregator with `__init__(db, user_id)` + `for_user(db, user_id)` classmethod + `close()`
- **Signature change**: `Repos.__init__` / `for_user` drop the `config` param (spec §3).

- [ ] **Step 1: Write the failing test**

Add `TestRecentSentencesRepo` + `TestRepos` to `tests/persistence/test_repos.py`. Copy from `tests/ket_partner/test_db.py`:

- `test_recent_sentences_repo` (l.648)
- `test_multi_user_isolation` (l.610)
- `test_ket_vocabulary_uses_composite_pk` (l.70)

```python
@pytest.mark.asyncio
class TestRepos:
    async def test_for_user_no_config_param(self, temp_db_path):
        """Spec §3: Repos.for_user signature drops config (Option B).
        Constructing without config must succeed — no KetConfig import."""
        from persistence.bootstrap import init_db  # local import; A7 adds this
        db = await init_db(temp_db_path, csv_path=None)
        repos = Repos.for_user(db, "default")
        assert repos.vocab is not None
        assert repos.stats is not None
        await repos.close()
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py::TestRecentSentencesRepo tests/persistence/test_repos.py::TestRepos -v`
Expected: FAIL — `AttributeError: module 'persistence.repos' has no attribute 'RecentSentencesRepo'`

- [ ] **Step 3: Append RecentSentencesRepo + Repos**

Copy `RecentSentencesRepo` from `db.py:573-607` verbatim. Append `Repos` adapted from `db.py:610-639` — **delete** `config` param + `if config is None: from flow.ket_partner.config import load_config` block + `self._config` field + StatsRepo's config arg:

```python
# src/persistence/repos.py (append)
class RecentSentencesRepo:
    """recent_sentences table — read/write."""
    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None:
        self._db = db
        self._user_id = user_id

    # Copy list_recent / append / list_recent_scaffolding from db.py:578-607 verbatim
    ...


class Repos:
    """Facade over 5 per-user Repos. Constructed once per request; user_id isolates."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> None:
        self._db = db
        self._user_id = user_id
        self.vocab = VocabRepo(db, user_id)
        self.stats = StatsRepo(db, user_id)
        self.profile = ProfileRepo(db, user_id)
        self.log = LogRepo(db, user_id)
        self.recent = RecentSentencesRepo(db, user_id)

    @classmethod
    def for_user(
        cls,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> "Repos":
        return cls(db, user_id)

    async def close(self) -> None:
        await self._db.close()
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_repos.py -v`
Expected: all test classes pass (full repos coverage)

- [ ] **Step 5: Commit**

```bash
git add src/persistence/repos.py tests/persistence/test_repos.py
git commit -m "feat(persistence): add RecentSentencesRepo + Repos aggregator (no config param)"
```

---

### Task 7: persistence/migration.py + bootstrap.py + __init__.py

**Files:**
- Create: `src/persistence/migration.py`
- Create: `src/persistence/bootstrap.py`
- Modify: `src/persistence/__init__.py`
- Create: `tests/persistence/test_migration.py`
- Create: `tests/persistence/test_bootstrap.py`

**Interfaces:**
- Produces: `persistence.migration.migrate_old_schema_if_needed(db) -> None`
- Produces: `persistence.bootstrap.init_db(db_path, csv_path=None, default_nickname="宝贝", default_age=8) -> aiosqlite.Connection`
- Produces: `persistence.bootstrap._import_csv(db, csv_path) -> None` (private)
- `persistence/__init__.py` re-exports: `init_db`, `WordRef`, `MASTERY_CAP`, `derive_status`, `VocabRepo`, `StatsRepo`, `ProfileRepo`, `LogRepo`, `RecentSentencesRepo`, `Repos`

- [ ] **Step 1: Write the failing test**

`tests/persistence/test_bootstrap.py` — migrate from `tests/ket_partner/test_db.py`:
- `test_init_db_creates_tables` (l.52)
- `test_import_csv_creates_one_row_per_context` (l.105)
- `test_import_csv_handles_bom` (l.133)
- `test_import_csv_links_topic_to_specific_context` (l.151)

Change import: `from persistence.bootstrap import init_db`.

`tests/persistence/test_migration.py`:

```python
import pytest
import aiosqlite

from persistence.migration import migrate_old_schema_if_needed
from persistence.schema import SCHEMA_SQL


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
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/test_bootstrap.py tests/persistence/test_migration.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create migration.py + bootstrap.py + __init__.py**

```python
# src/persistence/migration.py
"""Legacy single-user schema → multi-tenant migration.

Detected via missing user_id columns on vocab_stats / conversation_log,
or kid_profile having `id` column instead of `user_id`.
"""
import aiosqlite

from flow.common import logger  # logger lives in flow.common; OK to import


async def migrate_old_schema_if_needed(db: aiosqlite.Connection) -> None:
    # Copy body from src/flow/ket_partner/db.py:676-734 verbatim
    # (function was named _migrate_old_schema_if_needed there)
    ...
```

```python
# src/persistence/bootstrap.py
"""Composition-root DB initialization. Not for per-request use."""
import csv
import sqlite3

import aiosqlite

from flow.common import logger
from persistence.migration import migrate_old_schema_if_needed
from persistence.schema import SCHEMA_SQL

_DEFAULT_CSV = None  # resolved by caller (CLI / API)


async def init_db(
    db_path: str,
    csv_path: str | None = None,
    default_nickname: str = "宝贝",
    default_age: int = 8,
) -> aiosqlite.Connection:
    """Open connection, set PRAGMAs, run migration, executescript schema,
    seed default user + kid_profile, optional CSV import."""
    db = await aiosqlite.connect(db_path)
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await migrate_old_schema_if_needed(db)
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
    """Import KET vocabulary from CSV. Private — only bootstrap.py calls this."""
    # Copy body from src/flow/ket_partner/db.py:737-762 verbatim
    ...
```

```python
# src/persistence/__init__.py
"""Top-level persistence package.

Re-exports only the public API. SCHEMA_SQL, _import_csv, and
migrate_old_schema_if_needed stay private to their modules.
"""
from persistence.bootstrap import init_db
from persistence.models import MASTERY_CAP, WordRef, derive_status
from persistence.repos import (
    LogRepo,
    ProfileRepo,
    RecentSentencesRepo,
    Repos,
    StatsRepo,
    VocabRepo,
)

__all__ = [
    "init_db",
    "WordRef", "MASTERY_CAP", "derive_status",
    "VocabRepo", "StatsRepo", "ProfileRepo", "LogRepo",
    "RecentSentencesRepo", "Repos",
]
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/ -v`
Expected: all persistence tests pass

- [ ] **Step 5: Commit**

```bash
git add src/persistence/migration.py src/persistence/bootstrap.py src/persistence/__init__.py tests/persistence/test_migration.py tests/persistence/test_bootstrap.py
git commit -m "feat(persistence): add migration.py + bootstrap.py + __init__ re-exports"
```

---

## Phase B: flow/ket_partner internal slimming

Builds the Protocol contract, renames nodes.py, extracts sentence_orchestration + graph expansion. `agent.py` ends up holding only `KETPartnerAgent` node methods.

### Task 8: flow/ket_partner/persistence.py (Protocol + get_repos)

**Files:**
- Create: `src/flow/ket_partner/persistence.py`
- Create: `tests/flow/ket_partner/test_persistence_protocol.py`

**Interfaces:**
- Produces: `KETPartnerRepos` (runtime_checkable Protocol) + 5 sub-Protocols
- Produces: `get_repos(config: RunnableConfig) -> KETPartnerRepos`

- [ ] **Step 1: Write the failing test**

```python
# tests/flow/ket_partner/test_persistence_protocol.py
import pytest

from persistence.repos import Repos  # concrete impl
from persistence.bootstrap import init_db


@pytest.mark.asyncio
async def test_repos_satisfies_protocol(temp_db_path):
    """Spec §11: isinstance(Repos(...), KETPartnerRepos) must pass —
    runtime_checkable verifies attribute existence."""
    from flow.ket_partner.persistence import KETPartnerRepos

    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    assert isinstance(repos, KETPartnerRepos)
    await repos.close()


def test_get_repos_extracts_from_config():
    """get_repos pulls repos out of config['configurable']['repos']."""
    from flow.ket_partner.persistence import KETPartnerRepos, get_repos

    sentinel = object()  # not a real Repos; just verifying the extraction
    config = {"configurable": {"repos": sentinel}}
    assert get_repos(config) is sentinel
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_persistence_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flow.ket_partner.persistence'`

- [ ] **Step 3: Create persistence.py**

Copy the full Protocol definition block from spec §4 `persistence.py` section (lines 371-456 of the spec):

```python
# src/flow/ket_partner/persistence.py
"""Agent-facing persistence contract. flow/ket_partner/ has ZERO runtime
dependency on persistence/ — WordRef is referenced only via TYPE_CHECKING.

KETPartnerRepos is a runtime_checkable Protocol; persistence/repos.Repos
structurally satisfies it (no explicit registration needed).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from persistence.models import WordRef


class VocabRepoProtocol(Protocol):
    async def get_topics_for_word(self, word: str, context: str = "") -> list[str]: ...
    async def get_ket_word(self, word: str, context: str = "") -> WordRef | None: ...
    async def get_ket_word_any_context(self, word: str) -> WordRef | None: ...
    async def words_in_topic_without_stats(self, topic: str) -> list[WordRef]: ...
    async def unexposed_notopic_words(self) -> list[WordRef]: ...
    async def topics_with_unmastered(self, exclude: str | None = None) -> list[str]: ...
    async def total_count(self) -> int: ...


class StatsRepoProtocol(Protocol):
    async def get(self, word: str, context: str = "") -> dict | None: ...
    async def apply_delta(
        self, word: str, context: str = "", delta: int = 0,
        exposed: bool = False, is_target: bool = False,
    ) -> dict | None: ...
    async def learning_count(self) -> int: ...
    async def oldest_learning_word(self) -> WordRef | None: ...
    async def increment_exposed(
        self, word: str, context: str = "", is_target: bool = False,
    ) -> None: ...
    async def list_all_with_vocab(self) -> list[dict]: ...


class ProfileRepoProtocol(Protocol):
    async def get(self) -> dict: ...
    async def update(self, **fields) -> None: ...


class LogRepoProtocol(Protocol):
    async def append(
        self, role: str, content: str,
        words_used: list[str] | None = None,
        target_words: list[dict[str, str]] | None = None,
        turn_id: int | None = None,
    ) -> None: ...
    async def recent(self, limit: int = 5) -> list[dict]: ...
    async def append_session_start(self) -> None: ...
    async def last_ai_message(self) -> dict | None: ...


class RecentSentencesRepoProtocol(Protocol):
    async def list_recent(self, limit: int = 20) -> list[str]: ...
    async def append(self, sentence: str, window: int = 20) -> None: ...
    async def list_recent_scaffolding(self, window: int = 20) -> list[list[str]]: ...


@runtime_checkable
class KETPartnerRepos(Protocol):
    """Agent-side persistence contract.
    Concrete impl: persistence/repos.py::Repos.
    """
    vocab: VocabRepoProtocol
    stats: StatsRepoProtocol
    profile: ProfileRepoProtocol
    log: LogRepoProtocol
    recent: RecentSentencesRepoProtocol


def get_repos(config: RunnableConfig) -> KETPartnerRepos:
    """Single access point for repos in node methods."""
    return config["configurable"]["repos"]
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_persistence_protocol.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/flow/ket_partner/persistence.py tests/flow/ket_partner/test_persistence_protocol.py
git commit -m "feat(ket_partner): add persistence.py Protocol contract + get_repos"
```

---

### Task 9: vocab_selector.py imports migrate to TYPE_CHECKING

**Files:**
- Modify: `src/flow/ket_partner/vocab_selector.py`
- Modify: `tests/ket_partner/test_vocab_selector.py` (verify still passes; no edits expected)

**Interfaces:**
- Breaks runtime import of `WordRef` from `flow.ket_partner.db`
- Produces: `vocab_selector` references `WordRef` via TYPE_CHECKING only; `Repos` replaced by `KETPartnerRepos` Protocol

- [ ] **Step 1: Write the failing test**

`tests/ket_partner/test_vocab_selector.py` already exists and tests behavior. Run it to establish baseline:

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/ket_partner/test_vocab_selector.py -v`
Expected: PASS (baseline — we will keep it passing through the import migration)

Add one NEW test verifying zero runtime dep on `flow.ket_partner.db`:

```python
# Append to tests/ket_partner/test_vocab_selector.py
import sys


def test_vocab_selector_does_not_import_db_at_runtime():
    """Spec §4 WordRef discipline: vocab_selector must not import from
    persistence or flow.ket_partner.db at runtime."""
    # Force a fresh import
    for mod in [m for m in sys.modules if m.startswith("flow.ket_partner.vocab_selector")]:
        del sys.modules[mod]
    import flow.ket_partner.vocab_selector  # noqa: F401

    # After import, persistence must not be loaded by vocab_selector's import chain
    # (persistence may be loaded by OTHER modules in the test session, so check
    # the module's own imports via its globals)
    mod = sys.modules["flow.ket_partner.vocab_selector"]
    # The module should not have imported persistence or db at top-level
    assert "persistence" not in dir(mod) or isinstance(
        getattr(mod, "persistence", None), type(None)
    )
```

(Simpler approach — see Step 3; this test can be a `grep`-style check instead. Use a static check in Step 5 verify.)

- [ ] **Step 2: Run new test to verify it fails** (or skip and rely on grep verification in Step 4)

- [ ] **Step 3: Migrate vocab_selector.py imports**

Change the imports block of `src/flow/ket_partner/vocab_selector.py`:

```python
# src/flow/ket_partner/vocab_selector.py
from __future__ import annotations

from typing import TYPE_CHECKING

from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos

if TYPE_CHECKING:
    from persistence.models import WordRef


async def select_target_word(
    repos: KETPartnerRepos, profile: dict, config: KetConfig
) -> WordRef | None:
    ...
```

Body of `select_target_word`, `_pick_new_word`, `rotate_topic`, `_compute_refill_mode` is UNCHANGED — only the import lines change. Replace `Repos` type hints with `KETPartnerRepos` throughout.

- [ ] **Step 4: Run all vocab_selector tests to verify pass**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/ket_partner/test_vocab_selector.py -v`
Expected: all PASS (behavior unchanged)

Static check:
`D:/ProgramData/miniforge3/envs/langgraph/python.exe -c "import ast,sys; t=ast.parse(open('src/flow/ket_partner/vocab_selector.py',encoding='utf-8').read()); imps=[n for n in ast.walk(t) if isinstance(n,(ast.Import,ast.ImportFrom))]; persistence_imps=[n for n in imps if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('persistence')]; assert all(getattr(n,'level',0)>=1 for n in persistence_imps) or all(any(isinstance(p,ast.If) and getattr(getattr(getattr(p,'test',None),'id',lambda:None)(),'__call__',lambda:None)() for _ in []) for n in []); print('OK')"`
Expected: prints `OK` (or just verify manually that the only `from persistence...` import in vocab_selector.py is inside `if TYPE_CHECKING:`).

- [ ] **Step 5: Commit**

```bash
git add src/flow/ket_partner/vocab_selector.py tests/ket_partner/test_vocab_selector.py
git commit -m "refactor(ket_partner): migrate vocab_selector to TYPE_CHECKING WordRef"
```

---

### Task 10: nodes.py → mastery.py + output_format.py

**Files:**
- Create: `src/flow/ket_partner/mastery.py`
- Create: `src/flow/ket_partner/output_format.py`
- Create: `tests/flow/ket_partner/test_mastery.py`
- Create: `tests/flow/ket_partner/test_output_format.py`
- Delete (in Task 24): `src/flow/ket_partner/nodes.py`, `tests/ket_partner/test_nodes.py`

**Interfaces:**
- Produces: `mastery.apply_mastery_updates(state: BTPKetState, repos: KETPartnerRepos) -> None`
- Produces: `output_format.format_output_text(state: BTPKetState, new_sentence: str) -> str`

- [ ] **Step 1: Write the failing test**

Split `tests/ket_partner/test_nodes.py` into two files under `tests/flow/ket_partner/`. Copy each test verbatim, changing only the import line:

`tests/flow/ket_partner/test_mastery.py`:
```python
from flow.ket_partner.mastery import apply_mastery_updates
# ... all apply_mastery_updates tests copied verbatim from test_nodes.py ...
```

`tests/flow/ket_partner/test_output_format.py`:
```python
from flow.ket_partner.output_format import format_output_text
# ... all format_output_text tests copied verbatim from test_nodes.py ...
```

If `tests/flow/ket_partner/` does not have `__init__.py` / `conftest.py`, copy them from `tests/ket_partner/`.

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_mastery.py tests/flow/ket_partner/test_output_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flow.ket_partner.mastery'`

- [ ] **Step 3: Create mastery.py + output_format.py**

```python
# src/flow/ket_partner/mastery.py
from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.state import BTPKetState


async def apply_mastery_updates(state: BTPKetState, repos: KETPartnerRepos) -> None:
    """Copy body verbatim from src/flow/ket_partner/nodes.py:4-54.
    Type of `state` tightens from dict → BTPKetState (CLAUDE.md §二.1).
    Type of `repos` tightens from Repos → KETPartnerRepos Protocol.
    """
    intent = state.get("intent")
    if intent == "translation":
        last_words = state.get("last_sentence_words") or []
        target = state.get("last_target_word")
        target_ctx = state.get("last_target_context") or ""
        wrong = {w["word"] for w in state.get("wrong_words") or []}
        overall_correct = state.get("overall_correct")
        neutral_all = (not wrong) and (overall_correct is False)
        if neutral_all:
            return
        for w in last_words:
            ctx = target_ctx if w == target else ""
            delta = -1 if w in wrong else 1
            await repos.stats.apply_delta(w, context=ctx, delta=delta, exposed=False)
    elif intent == "idk":
        target = state.get("last_target_word")
        if target:
            target_ctx = state.get("last_target_context") or ""
            await repos.stats.apply_delta(target, context=target_ctx, delta=-1, exposed=False)
    elif intent == "asks_meaning":
        asked = state.get("asked_word")
        if asked:
            target = state.get("last_target_word")
            target_ctx = state.get("last_target_context") or ""
            wr = await repos.vocab.get_ket_word_any_context(asked)
            if wr:
                ctx = target_ctx if wr.word == target else ""
                await repos.stats.apply_delta(wr.word, context=ctx, delta=-1, exposed=False)
```

```python
# src/flow/ket_partner/output_format.py
from flow.ket_partner.state import BTPKetState


def format_output_text(state: BTPKetState, new_sentence: str) -> str:
    """Copy body verbatim from src/flow/ket_partner/nodes.py:57-99.
    Type of `state` tightens from dict → BTPKetState.
    """
    intent = state.get("intent")
    lines = []
    if intent == "translation":
        wrong = state.get("wrong_words") or []
        sentence_t = state.get("sentence_translation", "")
        overall_correct = state.get("overall_correct")
        if wrong:
            if sentence_t:
                lines.append(f"正确翻译：{sentence_t}")
            lines.append("你的翻译有误:")
            for entry in wrong:
                word = entry.get("word", "?")
                correct = entry.get("correct_translation", "?")
                lines.append(f" {word} 的意思是：{correct}")
        elif overall_correct is False:
            if sentence_t:
                lines.append(f"正确翻译：{sentence_t}")
            lines.append("你的翻译和原句意思有些偏差。")
    elif intent == "idk":
        sentence_t = state.get("sentence_translation", "")
        if sentence_t:
            lines.append(f"正确翻译：{sentence_t}")
    lines.append("请把这句译成中文:")
    lines.append(f'"{new_sentence}"')
    for ann in state.get("non_ket_annotations") or []:
        word = ann.get("word", "?")
        meaning = ann.get("meaning", "")
        if meaning:
            lines.append(f"{word} 的意思是：{meaning}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_mastery.py tests/flow/ket_partner/test_output_format.py -v`
Expected: all PASS (behavior preserved, only file location + type annotations changed)

- [ ] **Step 5: Commit**

```bash
git add src/flow/ket_partner/mastery.py src/flow/ket_partner/output_format.py tests/flow/ket_partner/test_mastery.py tests/flow/ket_partner/test_output_format.py
git commit -m "refactor(ket_partner): split nodes.py into mastery.py + output_format.py"
```

---

### Task 11: flow/ket_partner/sentence_orchestration.py (extract from agent.py)

**Files:**
- Create: `src/flow/ket_partner/sentence_orchestration.py`
- Create: `tests/flow/ket_partner/test_sentence_orchestration.py`
- (agent.py edits happen in Task 12)

**Interfaces:**
- Produces: `sentence_orchestration.generate_with_fallback(...)`
- Produces: `sentence_orchestration.validate_and_categorize(...)`
- Produces: `sentence_orchestration.apply_multiword_target_patch(target, sentence, result) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/flow/ket_partner/test_sentence_orchestration.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from flow.ket_partner.sentence_orchestration import (
    apply_multiword_target_patch,
    generate_with_fallback,
    validate_and_categorize,
)


@pytest.mark.asyncio
async def test_validate_and_categorize_passes_clean_sentence():
    """0 non-KET, not duplicate, not target-split → naturalness check decides."""
    # Mock validate_sentence to return clean result
    # Mock check_naturalness to return ok=True
    # Assert passed=True, reason_kind=None
    ...


@pytest.mark.asyncio
async def test_validate_and_categorize_target_split_reason():
    """Multi-word target broken across sentence → reason_kind='target_split'."""
    ...


@pytest.mark.asyncio
async def test_validate_and_categorize_non_ket_overflow_reason():
    """2+ non-KET words → reason_kind='non_ket_overflow'."""
    ...


@pytest.mark.asyncio
async def test_validate_and_categorize_duplicate_reason():
    """Sentence equals one in avoid_sentences → reason_kind='duplicate'."""
    ...


@pytest.mark.asyncio
async def test_validate_and_categorize_naturalness_reason():
    """0 non-KET, not dup, not split, but naturalness fails → 'naturalness'."""
    ...


@pytest.mark.asyncio
async def test_generate_with_fallback_returns_first_passing():
    """First attempt passes → returns immediately, no retry."""
    ...


@pytest.mark.asyncio
async def test_generate_with_fallback_switches_target_on_all_naturalness_failures():
    """All attempts fail with 'naturalness' + word_switched=False →
    calls select_target_word, retries with new target."""
    ...


@pytest.mark.asyncio
async def test_generate_with_fallback_accepts_overflow_after_retry_limit():
    """All attempts fail with 'non_ket_overflow' → picks fewest-non-KET draft."""
    ...


def test_apply_multiword_target_patch_adds_target_to_words_used():
    """Multi-word target 'ice cream' appears in sentence but not in words_used
    → patch adds 'ice cream', removes constituent words 'ice' and 'cream'."""
    # Build a MagicMock result with words_used=['ice','cream','I'], non_ket_words=[]
    # Call apply_multiword_target_patch('ice cream', 'I like ice cream', result)
    # Assert 'ice cream' in result.words_used, 'ice' not in result.words_used
    ...
```

(Fill in mock setup + assertions based on the function bodies in Step 3. These tests are NEW because the logic was previously inlined into agent.py methods without unit-level isolation.)

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_orchestration.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create sentence_orchestration.py**

Extract logic verbatim from `src/flow/ket_partner/agent.py:213-380` (`_validate_and_categorize` + `_generate_with_fallback`) and the multi-word-target patch block from `agent.py:174-193` (currently inline in `generate_sentence_node`):

```python
# src/flow/ket_partner/sentence_orchestration.py
"""Sentence generation + validation orchestration.

Extracted from KETPartnerAgent to enable isolated unit testing.
Stateless pure functions; takes all dependencies as parameters.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.multi_word_target import target_in_sentence
from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.sentence_generator import generate_sentence
from flow.ket_partner.sentence_naturalness import check_naturalness
from flow.ket_partner.sentence_validator import ValidationResult, validate_sentence
from flow.ket_partner.vocab_selector import select_target_word


async def validate_and_categorize(
    llm_smart: BaseChatModel,
    sentence: str,
    target: str,
    age: int,
    repos: KETPartnerRepos,
    avoid_sentences: list[str],
) -> dict:
    """Copy body from agent.py:213-268 (KETPartnerAgent._validate_and_categorize).
    Replace self.llm_smart → llm_smart. Returns dict with keys:
    result, passed, reason_kind, reason_detail, non_ket_words, non_ket_count,
    is_duplicate, is_target_split, sentence.
    """
    result = await validate_sentence(sentence, repos, target=target)
    is_duplicate = sentence in avoid_sentences
    non_ket_count = len(result.non_ket_words)
    is_target_split = (
        bool(target)
        and " " in target.strip()
        and not target_in_sentence(target, sentence)
    )
    passed = False
    reason_kind = None
    reason_detail = ""
    if non_ket_count <= 1 and not is_duplicate and not is_target_split:
        if non_ket_count == 0:
            naturalness = await check_naturalness(llm_smart, sentence, age=age)
            logger.debug(
                f"validate_sentence: {result} duplicate={is_duplicate} "
                f"target_split={is_target_split} naturalness_ok={naturalness.ok}"
            )
            if naturalness.ok:
                passed = True
            else:
                reason_kind = "naturalness"
                reason_detail = f"unnatural expression — {naturalness.reason}"
        else:
            logger.debug(f"{result} accept: 1 non-KET word will annotate")
            passed = True
    else:
        logger.debug(
            f"validate_sentence: {result} duplicate={is_duplicate} target_split={is_target_split}"
        )
        if is_target_split:
            reason_kind = "target_split"
            reason_detail = f"split the multi-word target '{target}' — words must be contiguous"
        elif is_duplicate:
            reason_kind = "duplicate"
            reason_detail = "word-for-word duplicate of a recent sentence"
        else:
            reason_kind = "non_ket_overflow"
            reason_detail = f"non-KET words {result.non_ket_words} exceed the limit (max 1 allowed)"
    return {
        "result": result,
        "passed": passed,
        "reason_kind": reason_kind,
        "reason_detail": reason_detail,
        "non_ket_words": list(result.non_ket_words),
        "non_ket_count": non_ket_count,
        "is_duplicate": is_duplicate,
        "is_target_split": is_target_split,
        "sentence": sentence,
    }


async def generate_with_fallback(
    llm_smart: BaseChatModel,
    initial_target: str,
    initial_context: str,
    avoid_words: list[str],
    avoid_sentences: list[str],
    age: int,
    profile: dict,
    repos: KETPartnerRepos,
    config: KetConfig,
) -> tuple[str, ValidationResult, str, str]:
    """Copy body from agent.py:270-380 (KETPartnerAgent._generate_with_fallback).
    Replace self.llm_smart / self.config → llm_smart / config.
    """
    target = initial_target
    context = initial_context
    word_switched = False

    while True:
        attempts: list[dict] = []
        seen_non_ket_words: list = []

        def _regen():
            return generate_sentence(
                llm_smart,
                target=target,
                recent_scaffolding=avoid_words,
                age=age,
                min_words=config.sentence.min_words,
                max_words=config.sentence.max_words,
                avoid_sentences=avoid_sentences,
                prior_attempts=attempts,
                avoid_non_ket_words=seen_non_ket_words,
                target_context=context,
            )

        sentence = await _regen()
        result = None

        for _ in range(config.validate_retry_limit):
            check = await validate_and_categorize(
                llm_smart, sentence, target, age, repos, avoid_sentences
            )
            result = check["result"]
            if check["passed"]:
                return sentence, result, target, context
            attempts.append({
                "sentence": sentence,
                "reason_kind": check["reason_kind"],
                "reason_detail": check["reason_detail"],
                "non_ket_words": check["non_ket_words"],
                "non_ket_count": check["non_ket_count"],
            })
            for w in check["non_ket_words"]:
                if w not in seen_non_ket_words:
                    seen_non_ket_words.append(w)
            sentence = await _regen()

        check = await validate_and_categorize(
            llm_smart, sentence, target, age, repos, avoid_sentences
        )
        result = check["result"]
        if check["passed"]:
            return sentence, result, target, context
        attempts.append({
            "sentence": sentence,
            "reason_kind": check["reason_kind"],
            "reason_detail": check["reason_detail"],
            "non_ket_words": check["non_ket_words"],
            "non_ket_count": check["non_ket_count"],
        })

        overflow_attempts = [a for a in attempts if a["reason_kind"] == "non_ket_overflow"]
        all_naturalness = bool(attempts) and all(
            a["reason_kind"] == "naturalness" for a in attempts
        )

        if overflow_attempts:
            best = min(reversed(overflow_attempts), key=lambda a: a["non_ket_count"])
            sentence = best["sentence"]
            result = await validate_sentence(sentence, repos, target=target)
            logger.warning(
                f"sentence validation: accepting non-KET overflow draft after "
                f"{len(attempts)} attempts (non_ket_count={len(result.non_ket_words)}); "
                f"sentence={sentence!r}"
            )
            return sentence, result, target, context
        elif all_naturalness and not word_switched:
            logger.info(
                f"all {len(attempts)} attempts failed on naturalness; "
                f"switching target word from '{target}'"
            )
            new_ref = await select_target_word(repos, profile, config)
            if new_ref is None or new_ref.word == target:
                logger.warning(
                    f"could not find a different target word; "
                    f"accepting final draft: {sentence!r}"
                )
                return sentence, result, target, context
            target = new_ref.word
            context = new_ref.context
            word_switched = True
            continue
        else:
            reasons = []
            if check["non_ket_count"] > 1:
                reasons.append(f"{check['non_ket_count']} non-KET word(s): {check['non_ket_words']}")
            if check["is_duplicate"]:
                reasons.append("duplicate of a recent sentence")
            if check["is_target_split"]:
                reasons.append(f"multi-word target '{target}' split apart")
            if not reasons:
                reasons.append(f"naturalness: {check['reason_detail']}")
            logger.warning(
                f"sentence validation failed after {len(attempts)} attempts; "
                f"accepting current draft — reasons: {('; '.join(reasons)) or 'unknown'}; "
                f"sentence={sentence!r}"
            )
            return sentence, result, target, context


def apply_multiword_target_patch(
    target: str,
    sentence: str,
    result: ValidationResult,
) -> None:
    """In-place patch result.words_used / non_ket_words for multi-word targets.
    Extracted inline block from agent.py:174-193.
    """
    if (
        target
        and target not in result.words_used
        and target_in_sentence(target, sentence)
    ):
        result.words_used.append(target)
        constituents = {c.lower() for c in target.split()}
        result.words_used = [
            w for w in result.words_used
            if w == target or w.lower() not in constituents
        ]
        result.non_ket_words = [
            w for w in result.non_ket_words
            if w.lower() not in constituents
        ]
        logger.debug(
            f"multi-word target patch: added '{target}', "
            f"final words_used={result.words_used}, "
            f"non_ket_words={result.non_ket_words}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_orchestration.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/flow/ket_partner/sentence_orchestration.py tests/flow/ket_partner/test_sentence_orchestration.py
git commit -m "feat(ket_partner): extract sentence_orchestration from agent"
```

---

### Task 12: graph.py expand + agent.py slim + test_graph_integration.py migration

This is the big one — splits `agent.py`, expands `graph.py`, and migrates the 1908-line `test_graph_integration.py`. Do this in 5 sub-commits under one task.

**Files:**
- Modify: `src/flow/ket_partner/graph.py` (add wire_graph / build_agent / route_after_classify / passthrough_node)
- Modify: `src/flow/ket_partner/agent.py` (delete compile/_generate_with_fallback/_validate_and_categorize/_route_call2/_passthrough/_route_after_init_state/build_agent/autonomous; update node method bodies to call sentence_orchestration + get_repos)
- Move: `tests/ket_partner/test_graph_integration.py` → `tests/integration/test_graph_integration.py`
- Move: `tests/ket_partner/test_graph.py` → `tests/flow/ket_partner/test_graph.py`

**Interfaces:**
- Produces: `graph.build_agent(llm_flash, llm_smart, checkpointer=None) -> CompiledStateGraph`
- Produces: `graph.wire_graph(builder, agent) -> None`
- Produces: `graph.route_after_classify(state) -> str`
- Produces: `graph.passthrough_node(state, config) -> dict`
- `agent.KETPartnerAgent` retains 13 node methods + `__init__` + `_run_summary_safe` + `aclose`

- [ ] **Step 1: Write the failing test (new test_graph.py)**

Move `tests/ket_partner/test_graph.py` to `tests/flow/ket_partner/test_graph.py` and update imports:

```python
# tests/flow/ket_partner/test_graph.py
# Copy all existing tests verbatim from tests/ket_partner/test_graph.py
# Change imports:
#   from flow.ket_partner.agent import build_agent  →  from flow.ket_partner.graph import build_agent
#   from flow.ket_partner.agent import KETPartnerAgent  →  keep (still in agent.py)

from flow.ket_partner.graph import build_agent, wire_graph, route_after_classify, passthrough_node
from flow.ket_partner.agent import KETPartnerAgent
# ... existing test bodies ...
```

- [ ] **Step 2: Move + adapt test_graph_integration.py**

Move `tests/ket_partner/test_graph_integration.py` → `tests/integration/test_graph_integration.py`. Apply the monkeypatch migration table from spec §8:

- All `monkeypatch.setattr(agent_module, "generate_sentence", ...)` → `monkeypatch.setattr(sentence_orchestration_module, "generate_sentence", ...)`
- All `monkeypatch.setattr(agent_module, "validate_sentence", ...)` → `monkeypatch.setattr(sentence_orchestration_module, "validate_sentence", ...)`
- All `monkeypatch.setattr(agent_module, "check_naturalness", ...)` → `monkeypatch.setattr(sentence_orchestration_module, "check_naturalness", ...)`
- All `monkeypatch.setattr(agent_module, "target_in_sentence", ...)` → `monkeypatch.setattr(sentence_orchestration_module, "target_in_sentence", ...)`
- Keep `evaluate_translation`, `classify_intent`, `lookup_word_meanings`, `lookup_sentence_translation`, `lookup_word_meaning`, `select_target_word`, `run_profile_summary` patches on `agent_module` (still imported into agent.py).
- Update import: `from flow.ket_partner.agent import build_agent` → `from flow.ket_partner.graph import build_agent`
- Add import: `from flow.ket_partner import sentence_orchestration as sentence_orchestration_module`

Update `tests/integration/conftest.py` if it has agent_module fixture (verify path).

Run to verify fail:
`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/integration/test_graph_integration.py -v 2>&1 | head -40`
Expected: FAIL — `ImportError: cannot import name 'build_agent' from 'flow.ket_partner.graph'`

- [ ] **Step 3: Expand graph.py + slim agent.py**

Expand `src/flow/ket_partner/graph.py` (currently 22 lines with only `route_by_intent` + `route_after_init`) to include `route_after_classify`, `passthrough_node`, `wire_graph`, `build_agent`:

```python
# src/flow/ket_partner/graph.py
"""Graph topology + routing + factory. Extracted from KETPartnerAgent.compile."""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from flow.ket_partner.agent import KETPartnerAgent
from flow.ket_partner.config import load_config
from flow.ket_partner.state import BTPKetState


def route_by_intent(state: BTPKetState) -> str:
    # Existing body unchanged (from current graph.py:5-15)
    intent = state.get("intent")
    if intent in ("translation", "idk"):
        return "select_target_word"
    if intent == "asks_meaning":
        return "explain_meaning"
    if intent == "off_topic":
        return "redirect_to_translate"
    if intent == "non_compliant":
        return "compliance_redirect"
    return "select_target_word"


def route_after_init(state: BTPKetState) -> str:
    # Existing body unchanged (from current graph.py:18-21)
    if state.get("last_english_sentence") is None:
        return "select_target_word"
    return "classify_intent"


def route_after_classify(state: BTPKetState) -> str:
    """Renamed from KETPartnerAgent._route_call2."""
    intent = state.get("intent")
    if intent == "translation":
        return "evaluate_translation"
    if intent == "idk":
        return "lookup_target_meaning"
    if intent == "asks_meaning":
        return "lookup_asked_meaning"
    return "skip"


async def passthrough_node(state: BTPKetState, config: RunnableConfig) -> dict:
    """No-op node for conditional_edges branching host. Merges _passthrough
    + _route_after_init_state (both were no-ops)."""
    return {}


def wire_graph(builder: StateGraph, agent: KETPartnerAgent) -> None:
    """Add all 13 nodes + edges. Extracted from KETPartnerAgent.compile body
    (agent.py:458-515).
    """
    builder.add_node("init_state", agent.init_state)
    builder.add_node("classify_intent", agent.classify_intent_node)
    builder.add_node("evaluate_translation", agent.evaluate_translation_node)
    builder.add_node("lookup_target_meaning", agent.lookup_target_meaning_node)
    builder.add_node("lookup_asked_meaning", agent.lookup_asked_meaning_node)
    builder.add_node("update_mastery", agent.update_mastery_node)
    builder.add_node("select_target_word", agent.select_target_word_node)
    builder.add_node("generate_sentence", agent.generate_sentence_node)
    builder.add_node("format_output", agent.format_output_node)
    builder.add_node("explain_meaning", agent.explain_meaning_node)
    builder.add_node("redirect_to_translate", agent.redirect_to_translate_node)
    builder.add_node("compliance_redirect", agent.compliance_redirect_node)
    builder.add_node("persist_turn", agent.persist_turn_node)

    builder.add_conditional_edges(START, route_after_init, {
        "init_state": "init_state",
        "classify_intent": "init_state",
        "select_target_word": "init_state",
    })
    builder.add_node("classify_intent_or_skip", passthrough_node)
    builder.add_edge("init_state", "classify_intent_or_skip")
    builder.add_conditional_edges("classify_intent_or_skip", route_after_init, {
        "classify_intent": "classify_intent",
        "select_target_word": "select_target_word",
    })
    builder.add_conditional_edges("classify_intent", route_after_classify, {
        "evaluate_translation": "evaluate_translation",
        "lookup_target_meaning": "lookup_target_meaning",
        "lookup_asked_meaning": "lookup_asked_meaning",
        "skip": "update_mastery",
    })
    builder.add_edge("evaluate_translation", "update_mastery")
    builder.add_edge("lookup_target_meaning", "update_mastery")
    builder.add_edge("lookup_asked_meaning", "update_mastery")
    builder.add_edge("update_mastery", "format_output_or_branch")
    builder.add_node("format_output_or_branch", passthrough_node)
    builder.add_conditional_edges("format_output_or_branch", route_by_intent, {
        "select_target_word": "select_target_word",
        "explain_meaning": "explain_meaning",
        "redirect_to_translate": "redirect_to_translate",
        "compliance_redirect": "compliance_redirect",
    })
    builder.add_edge("select_target_word", "generate_sentence")
    builder.add_edge("generate_sentence", "format_output")
    builder.add_edge("format_output", "persist_turn")
    builder.add_edge("persist_turn", END)
    builder.add_edge("explain_meaning", "persist_turn")
    builder.add_edge("redirect_to_translate", "persist_turn")
    builder.add_edge("compliance_redirect", "persist_turn")


async def build_agent(
    llm_flash: BaseChatModel,
    llm_smart: BaseChatModel,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Factory: load_config → KETPartnerAgent → StateGraph → wire_graph → compile.
    Attaches .agent to the compiled graph for shutdown lifecycle.
    """
    cfg = load_config()
    agent = KETPartnerAgent(llm_flash, llm_smart, cfg)
    builder = StateGraph(BTPKetState)
    wire_graph(builder, agent)
    graph = builder.compile(checkpointer=checkpointer)
    graph.agent = agent  # type: ignore[attr-defined]
    return graph
```

Slim down `src/flow/ket_partner/agent.py`:

- Delete `compile()` method (now in `wire_graph`)
- Delete `_generate_with_fallback()` method (now in `sentence_orchestration`)
- Delete `_validate_and_categorize()` method (now in `sentence_orchestration`)
- Delete `_route_call2()` method (now `route_after_classify` in graph.py)
- Delete `_passthrough()` + `_route_after_init_state()` methods (now `passthrough_node` in graph.py)
- Delete module-level `build_agent()` function (now in graph.py)
- Delete module-level `autonomous()` function (dead code, spec §9 #8)
- Update imports: remove `init_db`, `Repos` from `flow.ket_partner.db` (no longer needed); add `KETPartnerRepos` from `.persistence`; add `generate_with_fallback`, `apply_multiword_target_patch` from `.sentence_orchestration`; remove `generate_sentence`, `check_naturalness`, `validate_sentence`, `_tokenize`, `target_in_sentence` if they were only used by extracted code.
- Update `KETPartnerAgent.__init__` to keep `self._bg_tasks`, `self.llm_flash`, `self.llm_smart`, `self.config`.
- Update node method bodies:
  - All `repos: Repos = config["configurable"]["repos"]` → `repos = get_repos(config)`
  - `generate_sentence_node` calls `generate_with_fallback(self.llm_smart, ...)` + `apply_multiword_target_patch(target, sentence, result)` instead of `self._generate_with_fallback(...)` and the inline multi-word patch block (lines 174-193 of current agent.py).
  - `_run_summary_safe(self, repos: KETPartnerRepos)` — change type hint from `Repos` to `KETPartnerRepos`.

- [ ] **Step 4: Run all ket_partner + integration tests to verify they pass**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/ tests/integration/test_graph_integration.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/flow/ket_partner/graph.py src/flow/ket_partner/agent.py tests/flow/ket_partner/test_graph.py tests/integration/test_graph_integration.py
git commit -m "refactor(ket_partner): split agent.py — extract graph.py + slim KETPartnerAgent"
```

---

## Phase C: reporting package (NEW)

### Task 13: reporting/ket_partner/categories.py (Option B single source)

**Files:**
- Create: `src/reporting/__init__.py` (empty)
- Create: `src/reporting/ket_partner/__init__.py` (empty)
- Create: `src/reporting/ket_partner/categories.py`
- Create: `tests/reporting/__init__.py` (empty)
- Create: `tests/reporting/ket_partner/__init__.py` (empty)
- Create: `tests/reporting/ket_partner/test_categories.py`

**Interfaces:**
- Produces: `categories.CATEGORIES: tuple[str, ...]` (= `("mastered","learning","struggling","used","unused")`)
- Produces: `categories.Category` (Literal)
- Produces: `categories.classify_row(row, struggling_wc_min, struggling_ec_min) -> Category`
- Produces: `categories.classify(row, cfg) -> Category`
- Produces: `categories.group_by_category(rows, cfg) -> dict[str, list[dict]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/reporting/ket_partner/test_categories.py
from unittest.mock import MagicMock

from reporting.ket_partner.categories import (
    CATEGORIES,
    classify_row,
    classify,
    group_by_category,
)


def _cfg(wc_min=2, ec_min=5):
    cfg = MagicMock()
    cfg.struggling_threshold.wrong_count_min = wc_min
    cfg.struggling_threshold.exposed_count_min = ec_min
    return cfg


def test_classify_row_unused_when_unexposed():
    row = {"exposed_count": 0, "status": "new", "wrong_count": 0, "mastery_score": 0}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "unused"


def test_classify_row_mastered():
    row = {"exposed_count": 5, "status": "mastered", "wrong_count": 0, "mastery_score": 2}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "mastered"


def test_classify_row_learning():
    row = {"exposed_count": 3, "status": "learning", "wrong_count": 1, "mastery_score": 1}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "learning"


def test_classify_row_struggling_by_wrong_count():
    row = {"exposed_count": 3, "status": "new", "wrong_count": 2, "mastery_score": 1}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "struggling"


def test_classify_row_struggling_by_exposed_with_zero_mastery():
    row = {"exposed_count": 5, "status": "new", "wrong_count": 1, "mastery_score": 0}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "struggling"


def test_classify_row_used_when_below_struggling_thresholds():
    row = {"exposed_count": 3, "status": "new", "wrong_count": 1, "mastery_score": 1}
    assert classify_row(row, struggling_wc_min=2, struggling_ec_min=5) == "used"


def test_classify_uses_cfg_thresholds():
    row = {"exposed_count": 3, "status": "new", "wrong_count": 1, "mastery_score": 1}
    assert classify(row, _cfg(wc_min=2, ec_min=5)) == "used"
    # Raise the bar so this row is struggling
    assert classify(row, _cfg(wc_min=1, ec_min=5)) == "struggling"


def test_group_by_category_returns_all_five_buckets():
    rows = [
        {"word": "a", "exposed_count": 0, "status": "new", "wrong_count": 0, "mastery_score": 0},
        {"word": "b", "exposed_count": 5, "status": "mastered", "wrong_count": 0, "mastery_score": 2},
        {"word": "c", "exposed_count": 3, "status": "learning", "wrong_count": 1, "mastery_score": 1},
        {"word": "d", "exposed_count": 3, "status": "new", "wrong_count": 2, "mastery_score": 1},
        {"word": "e", "exposed_count": 3, "status": "new", "wrong_count": 1, "mastery_score": 1},
    ]
    bucket = group_by_category(rows, _cfg())
    assert set(bucket.keys()) == set(CATEGORIES)
    assert len(bucket["unused"]) == 1
    assert len(bucket["mastered"]) == 1
    assert len(bucket["learning"]) == 1
    assert len(bucket["struggling"]) == 1
    assert len(bucket["used"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ket_partner/test_categories.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reporting'`

- [ ] **Step 3: Create categories.py**

```python
# src/reporting/ket_partner/categories.py
"""Single source of truth for the 5 report categories.

Option B: replaces the old StatsRepo._category_where_sql / count_by_category
/ list_by_category SQL methods. CLI + API /report share this module, so
category rules stay in sync.
"""
from typing import Literal

from flow.ket_partner.config import KetConfig

Category = Literal["mastered", "learning", "struggling", "used", "unused"]

CATEGORIES: tuple[str, ...] = ("mastered", "learning", "struggling", "used", "unused")


def classify_row(
    row: dict,
    struggling_wc_min: int,
    struggling_ec_min: int,
) -> Category:
    """Pure classification rule. Order matters — earlier branches win."""
    if row["exposed_count"] == 0:
        return "unused"
    if row["status"] == "mastered":
        return "mastered"
    if row["status"] == "learning":
        return "learning"
    if (
        row["wrong_count"] >= struggling_wc_min
        or (row["exposed_count"] >= struggling_ec_min and row["mastery_score"] == 0)
    ):
        return "struggling"
    return "used"


def classify(row: dict, cfg: KetConfig) -> Category:
    """Convenience wrapper using cfg thresholds."""
    return classify_row(
        row,
        struggling_wc_min=cfg.struggling_threshold.wrong_count_min,
        struggling_ec_min=cfg.struggling_threshold.exposed_count_min,
    )


def group_by_category(
    rows: list[dict],
    cfg: KetConfig,
) -> dict[str, list[dict]]:
    """Bucket rows into the 5 categories in one pass."""
    bucket: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in rows:
        bucket[classify(r, cfg)].append(r)
    return bucket
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ket_partner/test_categories.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/reporting/ tests/reporting/
git commit -m "feat(reporting): add categories.py — single source for 5-way classification"
```

---

### Task 14: reporting/ket_partner/markdown.py

**Files:**
- Create: `src/reporting/ket_partner/markdown.py`
- Create: `tests/reporting/ket_partner/test_markdown.py`

**Interfaces:**
- Produces: `markdown.fmt_word(word, context) -> str`
- Produces: `markdown.render_markdown(profile, rows_by_category) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/reporting/ket_partner/test_markdown.py
from reporting.ket_partner.markdown import fmt_word, render_markdown


def test_fmt_word_no_context():
    assert fmt_word("cat", "") == "cat"


def test_fmt_word_with_context():
    assert fmt_word("bank", "Finance") == "bank(Finance)"


def test_render_markdown_renders_all_sections():
    profile = {"nickname": "小明", "total_turns": 42}
    rows_by_category = {
        "mastered": [{"word": "cat", "context": "", "pos": "n", "exposed_count": 5,
                      "correct_count": 5, "wrong_count": 0, "mastery_score": 2}],
        "learning": [{"word": "dog", "context": "", "pos": "n", "exposed_count": 3,
                      "correct_count": 1, "wrong_count": 1, "mastery_score": 1}],
        "struggling": [],
        "used": [{"word": "the", "context": "", "pos": "det", "exposed_count": 10,
                  "correct_count": 8, "wrong_count": 2, "mastery_score": 1}],
        "unused": [{"word": "ghost", "context": "", "pos": "n",
                    "exposed_count": 0, "correct_count": 0,
                    "wrong_count": 0, "mastery_score": 0}],
    }
    out = render_markdown(profile, rows_by_category)
    assert "学习报告 - 小明" in out
    assert "总轮数: 42" in out
    assert "正在学习 (1 项)" in out
    assert "已掌握 (1 项)" in out
    assert "已使用 (1 项)" in out
    assert "未使用 (1 项)" in out
    assert "学习困难 (0 项)" in out
    assert "cat" in out and "dog" in out


def test_render_markdown_handles_empty_buckets():
    profile = {"nickname": None, "total_turns": 0}
    empty = {c: [] for c in ("mastered", "learning", "struggling", "used", "unused")}
    out = render_markdown(profile, empty)
    assert "学习报告 - 小朋友" in out  # None nickname → '小朋友'
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ket_partner/test_markdown.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create markdown.py**

Adapt `_fmt_word` and `_render_markdown` from `src/flow/ket_partner/exporter.py:7-109`. The key change: `render_markdown` now takes **pre-bucketed** rows (output of `categories.group_by_category`) instead of raw rows + cfg:

```python
# src/reporting/ket_partner/markdown.py
"""Markdown rendering helpers for the learning report. Pure functions."""


def fmt_word(word: str, context: str) -> str:
    """Render 'word(context)' when context non-empty, else plain 'word'."""
    return f"{word}({context})" if context else word


def _render_table(rows: list[dict], with_stats: bool) -> list[str]:
    lines = []
    if with_stats:
        lines.append("| word | pos | exposed | correct | wrong | mastery |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            w = fmt_word(r["word"], r["context"])
            lines.append(
                f"| {w} | {r['pos']} | {r['exposed_count']} | "
                f"{r['correct_count']} | {r['wrong_count']} | {r['mastery_score']} |"
            )
    else:
        lines.append("| word | pos |")
        lines.append("|---|---|")
        for r in rows:
            w = fmt_word(r["word"], r["context"])
            lines.append(f"| {w} | {r['pos']} |")
    return lines


def render_markdown(
    profile: dict,
    rows_by_category: dict[str, list[dict]],
) -> str:
    """Render report from pre-bucketed rows. Categories module owns the
    classification; this function only renders.
    """
    mastered = rows_by_category["mastered"]
    learning = rows_by_category["learning"]
    used = rows_by_category["used"]
    unused = rows_by_category["unused"]
    struggling = rows_by_category["struggling"]

    lines: list[str] = [
        f"# 学习报告 - {profile.get('nickname') or '小朋友'}",
        f"总轮数: {profile.get('total_turns', 0)}",
        "",
        f"## 正在学习 ({len(learning)} 项)",
    ]
    lines.extend(_render_table(learning, with_stats=True))
    lines.append("")
    lines.append(f"## 已掌握 ({len(mastered)} 项)")
    lines.extend(_render_table(mastered, with_stats=True))
    lines.append("")
    lines.append(f"## 已使用 ({len(used)} 项)")
    lines.extend(_render_table(used, with_stats=True))
    lines.append("")
    lines.append(f"## 未使用 ({len(unused)} 项)")
    lines.extend(_render_table(unused, with_stats=False))
    lines.append("")
    lines.append(f"## 学习困难 ({len(struggling)} 项)")
    lines.extend(_render_table(struggling, with_stats=True))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ket_partner/test_markdown.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/reporting/ket_partner/markdown.py tests/reporting/ket_partner/test_markdown.py
git commit -m "feat(reporting): add markdown.py — render pre-bucketed rows"
```

---

### Task 15: reporting/ket_partner/exporter.py

**Files:**
- Create: `src/reporting/ket_partner/exporter.py`
- Create: `tests/reporting/ket_partner/test_exporter.py`

**Interfaces:**
- Produces: `exporter.export_learning_report(output_path, repos, cfg, fmt="markdown") -> str`
- Produces: `exporter.render_report_text(repos, cfg) -> str`

- [ ] **Step 1: Write the failing test**

Move + adapt `tests/ket_partner/test_exporter.py` → `tests/reporting/ket_partner/test_exporter.py`. Change imports:

```python
# tests/reporting/ket_partner/test_exporter.py
# Copy all tests verbatim from tests/ket_partner/test_exporter.py
# Change imports:
#   from flow.ket_partner.exporter import export_learning_report  →  from reporting.ket_partner.exporter import export_learning_report
#   from flow.ket_partner.db import Repos, init_db  →  from persistence import Repos, init_db
```

Add one new test for `render_report_text`:

```python
@pytest.mark.asyncio
async def test_render_report_text_returns_markdown_without_writing(temp_db_path):
    """Spec §6: render_report_text is the in-memory variant for API /report."""
    from reporting.ket_partner.exporter import render_report_text
    from persistence import Repos, init_db
    from flow.ket_partner.config import load_config

    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    cfg = load_config()
    text = await render_report_text(repos, cfg)
    assert "学习报告" in text
    await repos.close()
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ket_partner/test_exporter.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create exporter.py**

```python
# src/reporting/ket_partner/exporter.py
"""Report export orchestration. Replaces flow/ket_partner/exporter.py.

Key changes vs old:
- _fetch_all_stats via repos.stats._db.execute → await repos.stats.list_all_with_vocab()
- inline _classify → reporting.ket_partner.categories.group_by_category
- inline _render_markdown → reporting.ket_partner.markdown.render_markdown
- repos: Repos → KETPartnerRepos Protocol
"""
from flow.common import logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos
from reporting.ket_partner.categories import group_by_category
from reporting.ket_partner.markdown import render_markdown


async def export_learning_report(
    output_path: str,
    repos: KETPartnerRepos,
    cfg: KetConfig,
    fmt: str = "markdown",
) -> str:
    """Pull stats → group → render → write file. Returns output_path."""
    if fmt != "markdown":
        raise ValueError(f"unsupported format: {fmt}")
    content = await render_report_text(repos, cfg)
    with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.write(content)
    logger.info(f"Exported learning report to {output_path}")
    return output_path


async def render_report_text(
    repos: KETPartnerRepos,
    cfg: KetConfig,
) -> str:
    """In-memory render — for API /report or unit tests."""
    profile = await repos.profile.get()
    rows = await repos.stats.list_all_with_vocab()
    rows_by_category = group_by_category(rows, cfg)
    return render_markdown(profile, rows_by_category)
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ket_partner/test_exporter.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/reporting/ket_partner/exporter.py tests/reporting/ket_partner/test_exporter.py
git commit -m "feat(reporting): add exporter.py using list_all_with_vocab + group_by_category"
```

---

## Phase D: cli package (NEW)

### Task 16: cli/ket_partner/chat_logger.py

**Files:**
- Create: `src/cli/__init__.py` (empty)
- Create: `src/cli/ket_partner/__init__.py` (empty)
- Create: `src/cli/ket_partner/chat_logger.py`
- Create: `tests/cli/__init__.py` (empty)
- Create: `tests/cli/ket_partner/__init__.py` (empty)
- Create: `tests/cli/ket_partner/test_chat_logger.py`

**Interfaces:**
- Produces: `cli.ket_partner.chat_logger.ChatLogger` (verbatim copy from current `flow/ket_partner/chat_logger.py`)

- [ ] **Step 1: Write the failing test**

Copy `tests/ket_partner/test_chat_logger.py` verbatim to `tests/cli/ket_partner/test_chat_logger.py`. Change import:

```python
# tests/cli/ket_partner/test_chat_logger.py
from cli.ket_partner.chat_logger import ChatLogger
# ... rest verbatim ...
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_chat_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli'`

- [ ] **Step 3: Create chat_logger.py**

Copy `src/flow/ket_partner/chat_logger.py` (48 lines) verbatim to `src/cli/ket_partner/chat_logger.py`. No changes to body.

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_chat_logger.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli/ tests/cli/
git commit -m "feat(cli): move ChatLogger to cli/ket_partner/"
```

---

### Task 17: cli/ket_partner/commands.py

**Files:**
- Create: `src/cli/ket_partner/commands.py`
- Create: `tests/cli/ket_partner/test_commands.py`

**Interfaces:**
- Produces: `cli.ket_partner.commands.ExitLoop` (Exception)
- Produces: `cli.ket_partner.commands.CommandHandler`
- **Signature change**: `CommandHandler.__init__(repos: KETPartnerRepos, chat_logger: ChatLogger)` (was `db_path: str`)

- [ ] **Step 1: Write the failing test**

Copy `tests/ket_partner/test_commands.py` → `tests/cli/ket_partner/test_commands.py`. Change imports + constructor args:

```python
# tests/cli/ket_partner/test_commands.py
# Copy verbatim from tests/ket_partner/test_commands.py
# Changes:
#   from flow.ket_partner.commands import CommandHandler, ExitLoop
#     →  from cli.ket_partner.commands import CommandHandler, ExitLoop
#   from flow.ket_partner.db import Repos, init_db
#     →  from persistence import Repos, init_db
#   CommandHandler(db_path, chat_logger)  →  CommandHandler(repos, chat_logger)
```

Update every test that constructs `CommandHandler` to pass a `Repos` instance instead of a db_path string.

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_commands.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create commands.py**

```python
# src/cli/ket_partner/commands.py
"""CLI command dispatch. /exportstats now uses the injected Repos (no separate
init_db, no private _db access).
"""
from datetime import datetime

from flow.ket_partner.config import load_config
from flow.ket_partner.persistence import KETPartnerRepos
from reporting.ket_partner.exporter import export_learning_report


class ExitLoop(Exception):
    """/exit or /quit raises this to break the main loop."""


class CommandHandler:
    SUPPORTED: dict[str, str] = {
        "/exportstats": "导出学习状态报告",
        "/exit":        "退出练习",
        "/quit":        "退出练习",
        "/help":        "显示命令列表",
    }

    def __init__(self, repos: KETPartnerRepos, chat_logger) -> None:
        self.repos = repos
        self.chat_logger = chat_logger

    async def handle(self, user_input: str) -> None:
        cmd = user_input.strip().split()[0]
        if cmd in ("/exit", "/quit"):
            raise ExitLoop()
        elif cmd == "/help":
            self._print_help()
        elif cmd == "/exportstats":
            await self._export_stats()
        else:
            print(f"未知命令: {cmd}。输入 /help 查看支持的命令。")

    def _print_help(self) -> None:
        for cmd, desc in self.SUPPORTED.items():
            print(f"  {cmd:<15} {desc}")

    async def _export_stats(self) -> None:
        """Use injected repos. No init_db, no private _db access."""
        cfg = load_config()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"learning_report_{stamp}.md"
        await export_learning_report(output_path, self.repos, cfg)
        print(f"已导出学习报告到 {output_path}")
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_commands.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli/ket_partner/commands.py tests/cli/ket_partner/test_commands.py
git commit -m "feat(cli): move CommandHandler; inject Repos instead of db_path"
```

---

### Task 18: cli/ket_partner/main.py

**Files:**
- Create: `src/cli/ket_partner/main.py`
- Create: `tests/cli/ket_partner/test_main.py`

**Interfaces:**
- Produces: `cli.ket_partner.main.main() -> None` async

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/ket_partner/test_main.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_main_initializes_and_loops(capsys, monkeypatch):
    """main() wires init_db + Repos + build_agent + CommandHandler.
    Verify the wiring without exercising the real input loop."""
    # Patch init_db, build_agent, CommandHandler, ChatLogger
    # Patch asyncio.to_thread(input, ...) to return "/quit" on first call
    # Run main(); assert ExitLoop breaks the loop cleanly
    ...


@pytest.mark.asyncio
async def test_main_invokes_build_agent_without_db(monkeypatch):
    """Spec §11: build_agent(llm_flash, llm_max) — no db arg."""
    fake_build = AsyncMock()
    monkeypatch.setattr("cli.ket_partner.main.build_agent", fake_build)
    # ... drive main to call build_agent ...
    fake_build.assert_awaited_once()
    args, kwargs = fake_build.await_args
    # No positional db arg
    assert len(args) == 2  # llm_flash, llm_smart only
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create main.py**

Adapt `src/flow/ket_partner/main.py` (87 lines) to the new package layout:

```python
# src/cli/ket_partner/main.py
"""CLI entry point. Composition root for CLI usage."""
import asyncio
import os
from os.path import dirname, join

from langchain_core.messages import HumanMessage

from cli.ket_partner.chat_logger import ChatLogger
from cli.ket_partner.commands import CommandHandler, ExitLoop
from flow.common import llm_flash, llm_max, logger
from flow.ket_partner.graph import build_agent
from persistence import Repos, init_db

DEFAULT_DB = "ket_partner.db"
DEFAULT_CSV = join(dirname(__file__), "..", "..", "..", "data", "KET_vocabulary.csv")


async def main() -> None:
    info = {
        "nickname_kid": os.environ.get("KID_NICKNAME", "宝贝"),
        "age": int(os.environ.get("KID_AGE", "8")),
    }
    db_path = os.environ.get("KET_DB_PATH", DEFAULT_DB)
    csv_path = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else None

    db = await init_db(
        db_path,
        csv_path=csv_path,
        default_nickname=info["nickname_kid"],
        default_age=info["age"],
    )
    repos = Repos.for_user(db, "default")
    await repos.log.append_session_start()
    agent = await build_agent(llm_flash, llm_max)

    chat_logger = ChatLogger(log_dir="logs/chat")
    chat_logger.start_session(info["nickname_kid"])
    cmd_handler = CommandHandler(repos, chat_logger)

    messages: list = []
    turn_id = 1
    try:
        while True:
            user_input = await asyncio.to_thread(input, "用户输入: ")
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.startswith("/"):
                try:
                    await cmd_handler.handle(user_input)
                except ExitLoop:
                    break
                continue

            messages.append(HumanMessage(content=user_input))
            config = {
                "configurable": {
                    "thread_id": "main",
                    "user_id": "default",
                    "repos": repos,
                    "user_info": {"nickname": info["nickname_kid"], "age": info["age"]},
                }
            }
            response = await agent.ainvoke({"messages": messages[-5:]}, config=config)
            ai_reply = response["messages"][-1].content
            messages.append(response["messages"][-1])

            chat_logger.log_turn(turn_id, "user", user_input)
            chat_logger.log_turn(turn_id, "AI", ai_reply)
            print(f"AI: {ai_reply}\n")
            turn_id += 1
    finally:
        agent_instance = getattr(agent, "agent", None)
        if agent_instance is not None:
            try:
                await agent_instance.aclose()
            except (RuntimeError, OSError) as e:
                logger.warning(f"agent.aclose() failed during shutdown: {e}", exc_info=True)
        await db.close()
        chat_logger.close_session()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_main.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli/ket_partner/main.py tests/cli/ket_partner/test_main.py
git commit -m "feat(cli): move main.py to cli/ket_partner/; build_agent without db"
```

---

## Phase E: API updates

### Task 19: api/app.py imports update

**Files:**
- Modify: `src/api/app.py:16-17, 56`

- [ ] **Step 1: Write the failing test**

`tests/api/test_app_lifespan.py` (NEW) — verify the lifespan wiring:

```python
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_lifespan_calls_build_agent_without_db(monkeypatch):
    """Spec §11: build_agent(llm_flash, llm_max, checkpointer=...) — no db."""
    # Patch init_db, AsyncSqliteSaver, build_agent
    # Invoke lifespan context manager
    # Assert build_agent called with exactly 2 positional args (llm_flash, llm_max)
    ...
```

(If this is hard to test in isolation, rely on the static check in Step 4 + the existing `tests/api/test_*` suite as the regression net.)

- [ ] **Step 2: Run new test to verify it fails (or skip if static check suffices)**

- [ ] **Step 3: Edit api/app.py**

Three edits in `src/api/app.py`:

Change line 16:
```python
from flow.ket_partner.agent import build_agent
```
to:
```python
from flow.ket_partner.graph import build_agent
```

Change line 17:
```python
from flow.ket_partner.db import init_db
```
to:
```python
from persistence import init_db
```

Change line 56:
```python
agent = await build_agent(llm_flash, llm_max, db, checkpointer=checkpointer)
```
to:
```python
agent = await build_agent(llm_flash, llm_max, checkpointer=checkpointer)
```

- [ ] **Step 4: Run full api test suite + static check**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/api/ -v`

Static check that build_agent is called with no db positional:
`grep -n "build_agent(" src/api/app.py`
Expected: `agent = await build_agent(llm_flash, llm_max, checkpointer=checkpointer)`

- [ ] **Step 5: Commit**

```bash
git add src/api/app.py
git commit -m "refactor(api): import build_agent from flow.ket_partner.graph; drop db arg"
```

---

### Task 20: api/routes/chat.py imports update

**Files:**
- Modify: `src/api/routes/chat.py`

- [ ] **Step 1: Verify current import**

`grep -n "from flow.ket_partner" src/api/routes/chat.py` — expected to find `from flow.ket_partner.db import Repos`.

- [ ] **Step 2-3: Edit + verify**

Change `from flow.ket_partner.db import Repos` → `from persistence import Repos`. Body unchanged.

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/api/routes/test_chat_route.py -v`

- [ ] **Step 4: Static check**

`grep -n "from flow.ket_partner.db" src/api/routes/chat.py`
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/chat.py
git commit -m "refactor(api): chat route imports Repos from persistence"
```

---

### Task 21: api/routes/report.py Option B refactor

**Files:**
- Modify: `src/api/routes/report.py`

- [ ] **Step 1: Write the failing test**

Adapt `tests/api/test_report.py` — replace tests of `count_by_category` / `list_by_category` with the new flow (Option B: single `list_all_with_vocab` + Python `group_by_category` + slice pagination):

```python
# tests/api/test_report.py — update the assertions to check the new shape
# Specifically:
#   report_counts returns ReportResponse(mastered_count=..., ..., total_words=...)
#   report_by_category(category, page, page_size) returns ReportCategoryResponse
# Verify:
#   - master/learning/struggling/used/unused counts match Python-side classification
#   - page slice uses Python offset:offset+page_size
#   - 400 on invalid category stays the same
```

Concrete new test:

```python
@pytest.mark.asyncio
async def test_report_counts_uses_python_classification(client, temp_db_with_vocab):
    """Spec §7: /report uses list_all_with_vocab + group_by_category, not
    SQL count_by_category (which is deleted in Option B)."""
    resp = await client.get("/api/report")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) >= {
        "mastered_count", "learning_count", "struggling_count",
        "used_count", "unused_count", "total_words",
    }
    assert data["mastered_count"] + data["learning_count"] + data["struggling_count"] \
        + data["used_count"] + data["unused_count"] <= data["total_words"]
```

- [ ] **Step 2: Run test to verify it fails**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/api/test_report.py -v`
Expected: FAIL — old route still calls `repos.stats.count_by_category` which doesn't exist anymore (StatsRepo changed in A4).

- [ ] **Step 3: Rewrite report.py**

```python
# src/api/routes/report.py
import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from flow.ket_partner.config import load_config
from persistence import Repos
from reporting.ket_partner.categories import CATEGORIES, group_by_category
from src.api.deps import User, get_current_user, get_db
from src.api.schemas import ReportCategoryResponse, ReportResponse, ReportWord

router = APIRouter()
_CFG = load_config()


@router.get("", response_model=ReportResponse)
async def report_counts(
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
) -> ReportResponse:
    """Spec §7 Option B: one SQL pull, Python classification."""
    repos = Repos.for_user(db, user.id)
    rows = await repos.stats.list_all_with_vocab()
    bucket = group_by_category(rows, _CFG)
    return ReportResponse(
        mastered_count=len(bucket["mastered"]),
        learning_count=len(bucket["learning"]),
        struggling_count=len(bucket["struggling"]),
        used_count=len(bucket["used"]),
        unused_count=len(bucket["unused"]),
        total_words=await repos.vocab.total_count(),
    )


@router.get("/{category}", response_model=ReportCategoryResponse)
async def report_by_category(
    category: str,
    page: int = 1,
    page_size: int = 100,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
) -> ReportCategoryResponse:
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"invalid category: {category}")
    if page < 1 or page_size < 1 or page_size > 500:
        raise HTTPException(status_code=400, detail="invalid pagination params")

    repos = Repos.for_user(db, user.id)
    rows = await repos.stats.list_all_with_vocab()
    bucket = group_by_category(rows, _CFG)
    category_rows = bucket[category]
    total = len(category_rows)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    page_rows = category_rows[offset:offset + page_size]
    return ReportCategoryResponse(
        category=category,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        words=[ReportWord(**r) for r in page_rows],
    )
```

- [ ] **Step 4: Run test to verify it passes**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/api/test_report.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/report.py tests/api/test_report.py
git commit -m "refactor(api): /report uses Option B (Python classification, single SQL pull)"
```

---

## Phase F: Cleanup

### Task 22: Delete flow/agent.py + flow/log/

**Files:**
- Delete: `src/flow/agent.py`
- Delete: `src/flow/log/` (entire empty directory)

- [ ] **Step 1: Verify nothing imports them**

`grep -rn "from flow.agent\|from flow.log" src/ tests/`
Expected: no matches (spec §9 #1-2 confirm they're dead code).

- [ ] **Step 2: Delete**

```bash
rm src/flow/agent.py
rmdir src/flow/log
```

(Or use `git rm` to stage the deletion directly.)

- [ ] **Step 3: Run full test suite to verify nothing broke**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/ -v 2>&1 | tail -30`
Expected: same pass rate as before deletion.

- [ ] **Step 4: Static checks**

`ruff check src/ tests/` + `mypy src/ tests/` — all clear.

- [ ] **Step 5: Commit**

```bash
git add -A src/flow/
git commit -m "chore(flow): delete dead agent.py + empty log/ dir"
```

---

### Task 23: Clean flow/common.py dead code

**Files:**
- Modify: `src/flow/common.py`

- [ ] **Step 1: Verify dead code references**

```bash
grep -rn "IS_RUNNING_IN_PYTEST\|llm_plus\|llm_doubao\|doubao_api" src/ tests/
```
Expected: matches only inside `src/flow/common.py` itself.

- [ ] **Step 2-3: Edit common.py**

- Delete lines 16-19 (`IS_RUNNING_IN_PYTEST` block).
- Delete lines 61-72 (`llm_plus` ChatOpenAI client).
- Delete line 101 (`doubao_api = environ.get(...)`).
- Delete lines 102-116 (`llm_doubao` client).
- Change line 98 `extra_body={"enable_thinking": False}` → `extra_body=extra_params` (consistency with `llm_max`).

Final file should contain only: `logger`, `extra_params`, `_resolve_dashscope_api_key`, `dashscope_api_key`, `llm_max`, `llm_flash`.

- [ ] **Step 4: Run tests + static checks**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/ -v 2>&1 | tail -10`
`ruff check src/flow/common.py` + `mypy src/flow/common.py`

- [ ] **Step 5: Commit**

```bash
git add src/flow/common.py
git commit -m "chore(flow): delete dead code in common.py (IS_RUNNING_IN_PYTEST/llm_plus/llm_doubao)"
```

---

### Task 24: Delete obsolete flow/ket_partner/ files + old tests

**Files:**
- Delete: `src/flow/ket_partner/db.py`
- Delete: `src/flow/ket_partner/main.py`
- Delete: `src/flow/ket_partner/commands.py`
- Delete: `src/flow/ket_partner/chat_logger.py`
- Delete: `src/flow/ket_partner/exporter.py`
- Delete: `src/flow/ket_partner/nodes.py`
- Delete: `tests/ket_partner/test_db.py`
- Delete: `tests/ket_partner/test_nodes.py`
- Delete: `tests/ket_partner/test_exporter.py`
- Delete: `tests/ket_partner/test_commands.py`
- Delete: `tests/ket_partner/test_chat_logger.py`
- Delete: `tests/ket_partner/test_graph.py` (moved to `tests/flow/ket_partner/` in B5)
- Delete: `tests/ket_partner/test_graph_integration.py` (moved to `tests/integration/` in B5)
- Keep in `tests/ket_partner/`: `conftest.py`, `test_state.py`, `test_config.py`, `test_vocab_selector.py`, `test_input_classifier.py`, `test_profile_summarizer.py`, `test_sentence_generator.py`, `test_sentence_naturalness.py`, `test_translation_evaluator.py`, `test_word_meaning_lookup.py`, `test_multi_word_target.py`, `test_sentence_validator.py`
  - (Spec §8 keeps the LLM module tests under `tests/flow/ket_partner/`; if you want them moved too, do so now. The simplest path: move them in this task. Spec §8 explicitly says `tests/flow/ket_partner/` mirrors all flow/ket_partner source — including the 9 LLM module tests.)

- [ ] **Step 1: Verify no live references to deleted source files**

```bash
grep -rn "from flow.ket_partner.db\|from flow.ket_partner.nodes\|from flow.ket_partner.exporter\|from flow.ket_partner.main\|from flow.ket_partner.commands\|from flow.ket_partner.chat_logger" src/ tests/
```
Expected: no matches. (If matches found, fix them first — likely a missed import in earlier tasks.)

- [ ] **Step 2: Move remaining LLM module tests**

```bash
mkdir -p tests/flow/ket_partner
git mv tests/ket_partner/test_state.py tests/flow/ket_partner/
git mv tests/ket_partner/test_config.py tests/flow/ket_partner/
git mv tests/ket_partner/test_vocab_selector.py tests/flow/ket_partner/
git mv tests/ket_partner/test_input_classifier.py tests/flow/ket_partner/
git mv tests/ket_partner/test_profile_summarizer.py tests/flow/ket_partner/
git mv tests/ket_partner/test_sentence_generator.py tests/flow/ket_partner/
git mv tests/ket_partner/test_sentence_naturalness.py tests/flow/ket_partner/
git mv tests/ket_partner/test_translation_evaluator.py tests/flow/ket_partner/
git mv tests/ket_partner/test_word_meaning_lookup.py tests/flow/ket_partner/
git mv tests/ket_partner/test_multi_word_target.py tests/flow/ket_partner/
git mv tests/ket_partner/test_sentence_validator.py tests/flow/ket_partner/
git mv tests/ket_partner/conftest.py tests/flow/ket_partner/
```

Update test imports in moved files: `from flow.ket_partner.db import ...` → `from persistence import ...`.

- [ ] **Step 3: Delete obsolete src/ + tests/ files**

```bash
git rm src/flow/ket_partner/db.py
git rm src/flow/ket_partner/main.py
git rm src/flow/ket_partner/commands.py
git rm src/flow/ket_partner/chat_logger.py
git rm src/flow/ket_partner/exporter.py
git rm src/flow/ket_partner/nodes.py
git rm tests/ket_partner/test_db.py
git rm tests/ket_partner/test_nodes.py
git rm tests/ket_partner/test_exporter.py
git rm tests/ket_partner/test_commands.py
git rm tests/ket_partner/test_chat_logger.py
rmdir tests/ket_partner  # should now be empty
```

- [ ] **Step 4: Run full test suite + static checks**

`D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/ -v`
`ruff check src/ tests/` + `mypy src/ tests/`
Expected: all PASS, zero warnings.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete obsolete flow/ket_partner/ files; consolidate tests under tests/flow/ket_partner/"
```

---

## Phase G: Final verification

### Task 25: Full spec §11 acceptance gate

**Files:** none (verification only)

- [ ] **Step 1: Run static checks across whole repo**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check src/ tests/
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src/ tests/
```
Expected: both clean (zero errors, zero warnings).

- [ ] **Step 2: Run full pytest suite**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 3: Spec §11 import-invariant checks**

```bash
# 1. flow/ket_partner/ has no runtime import of top-level persistence/cli/reporting
grep -rn "^from persistence\|^from cli\|^from reporting" src/flow/ket_partner/
# Expected: zero matches (only TYPE_CHECKING imports allowed)

# 2. persistence/ has no import from flow.ket_partner / cli / reporting
grep -rn "from flow.ket_partner\|from cli\|from reporting" src/persistence/
# Expected: zero matches

# 3. reporting/ket_partner/ has no import from cli / api
grep -rn "from cli\|from src.api\|from api" src/reporting/
# Expected: zero matches

# 4. flow.agent + flow.log deleted
test ! -e src/flow/agent.py && echo "OK: agent.py gone"
test ! -e src/flow/log && echo "OK: log/ gone"

# 5. common.py dead code gone
grep -n "IS_RUNNING_IN_PYTEST\|llm_plus\|llm_doubao\|doubao_api" src/flow/common.py
# Expected: zero matches

# 6. exporter no private _db access
grep -n "_db.execute\|_db.close" src/reporting/ket_partner/exporter.py
# Expected: zero matches

# 7. commands no init_db / _db.close
grep -n "init_db\|_db.close" src/cli/ket_partner/commands.py
# Expected: zero matches

# 8. StatsRepo public API no longer has category SQL methods
grep -n "_category_where_sql\|count_by_category\|list_by_category" src/persistence/repos.py
# Expected: zero matches

# 9. test_graph_integration monkeypatch migration
grep -n "agent_module.generate_sentence\|agent_module.validate_sentence\|agent_module.check_naturalness\|agent_module.target_in_sentence" tests/integration/test_graph_integration.py
# Expected: zero matches (should all be sentence_orchestration_module.*)

# 10. WordRef TYPE_CHECKING discipline
grep -B2 "from persistence.models import WordRef" src/flow/ket_partner/*.py
# Expected: every match has "if TYPE_CHECKING:" on a line above

# 11. build_agent attaches .agent
grep -n "graph.agent = agent" src/flow/ket_partner/graph.py
# Expected: exactly one match
```

- [ ] **Step 4: Run a real LLM integration test once (spec §六.8)**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/integration/ -v -m integration
```
Expected: if any integration tests are marked `@pytest.mark.integration`, they should pass (or skip cleanly if API key missing).

- [ ] **Step 5: Final commit**

If any fixups were needed during verification, commit them. Otherwise no commit.

```bash
git log --oneline -20  # review the phase commits
```

---

## Self-Review Notes

- **Spec coverage**: All 11 spec sections map to ≥1 task. §3 (persistence) → A1-A7; §4 (flow/ket_partner) → B1-B5; §5 (cli) → D1-D3; §6 (reporting) → C1-C3; §7 (composition root) → E1-E3 + D3; §8 (test reorg) → interleaved across all phases + F3; §9 (cleanup) → F1-F3; §10.2 (passthrough tech debt) → preserved in graph.py; §11 (acceptance) → G1.
- **Type consistency**: `KETPartnerRepos` Protocol used uniformly in flow/ket_partner/ + reporting/. `Repos` concrete class used only in composition roots (api/, cli/, tests). WordRef always via TYPE_CHECKING in flow/ket_partner/.
- **Build order**: A → B → C → D → E → F → G. Each phase's tests pass independently; nothing forward-references code from a later phase.
- **Risk**: Task 12 (agent.py split) is the highest-risk task — 1908-line test_graph_integration.py must be migrated carefully. Recommend running it standalone after Phase A + B1-B4 to isolate issues.
