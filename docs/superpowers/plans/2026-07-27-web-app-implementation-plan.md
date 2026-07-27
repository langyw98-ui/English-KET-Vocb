# KET Partner Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the existing CLI KET Partner agent into a FastAPI + Vue 3 Web application allowing children to practice English via browser and parents to inspect learning reports.

**Architecture:** A FastAPI backend serves REST endpoints (`/api/chat`, `/api/report`, `/api/report/{category}`, `/api/messages`) backed by SQLite and a stateless LangGraph Agent using `AsyncSqliteSaver`. Production mode serves the built Vue 3 static bundle from `web/dist/` in a single uvicorn process, while development mode proxies `/api` via Vite dev server.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, aiosqlite, LangGraph, LangChain, Vue 3, Vite, TypeScript, Pinia, Vue Router 4, pytest, httpx.

## Global Constraints

- **Python Environment**: All Python/pytest invocations must use `D:/ProgramData/miniforge3/envs/langgraph/python.exe`.
- **RTK Command Rule**: All terminal shell commands must be prefixed with `rtk` (e.g. `rtk pytest`).
- **Code Standards**: Adhere strictly to `.agents/rules/code-standards.md` — explicit exception types (no bare `except`), `logger.warning(..., exc_info=True)` on fallbacks, no blocking IO in async functions, hermetic tests.
- **Structured Output**: All LLM structured outputs must specify `method="function_calling"`.
- **No Emoji**: Console and log outputs must not contain emoji due to Windows GBK encoding.

---

### Task 1: Database Schema & Repo Layer Refactoring (Multi-Tenant Support)

**Files:**
- Modify: `src/flow/ket_partner/db.py`
- Modify: `src/flow/ket_partner/exporter.py`
- Modify: `tests/ket_partner/test_db.py`

**Interfaces:**
- Consumes: Existing SQLite connection logic and `KetConfig` in `src/flow/ket_partner/config.py`
- Produces: Updated `_SCHEMA`, `init_db()`, `Repos.for_user(db, user_id)`, `RecentSentencesRepo`, `StatsRepo.count_by_category()` / `list_by_category()`

- [ ] **Step 1: Write failing tests for multi-tenant Repos and category queries**

Create/update tests in `tests/ket_partner/test_db.py` to test per-user isolation and category counts:

```python
import pytest
import aiosqlite
from flow.ket_partner.db import init_db, Repos
from flow.ket_partner.config import load_config

@pytest.mark.asyncio
async def test_multi_user_isolation(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = await init_db(db_path)
    try:
        repos_a = Repos.for_user(db, "user_a")
        repos_b = Repos.for_user(db, "user_b")
        
        await repos_a.stats.increment_exposed("cat", context="slipping")
        
        stats_a = await repos_a.stats.get("cat", context="slipping")
        stats_b = await repos_b.stats.get("cat", context="slipping")
        
        assert stats_a["exposed_count"] == 1
        assert stats_b is None
    finally:
        await db.close()

@pytest.mark.asyncio
async def test_stats_count_and_list_by_category(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = await init_db(db_path)
    cfg = load_config()
    try:
        repos = Repos(db, "default", cfg)
        await db.execute(
            "INSERT INTO ket_vocabulary (word, context, pos) VALUES ('cat', '', 'noun')"
        )
        await db.commit()
        
        unused_cnt = await repos.stats.count_by_category("unused")
        assert unused_cnt >= 1
    finally:
        await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk pytest tests/ket_partner/test_db.py -k test_multi_user_isolation`
Expected: FAIL due to missing `user_id` support and `for_user` method.

- [ ] **Step 3: Refactor `db.py` schema, `init_db`, `Repos`, `RecentSentencesRepo` and `StatsRepo`**

Update `_SCHEMA`, `init_db()`, `Repos`, `StatsRepo` in `src/flow/ket_partner/db.py`:

```python
# In src/flow/ket_partner/db.py:
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

async def init_db(
    db_path: str,
    csv_path: Optional[str] = None,
    default_nickname: str = "宝贝",
    default_age: int = 8,
) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.executescript(_SCHEMA)

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
```

Implement `RecentSentencesRepo` and `StatsRepo._category_where_sql` as specified in the Spec.

- [ ] **Step 4: Run tests to verify they pass**

Run: `rtk pytest tests/ket_partner/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/flow/ket_partner/db.py src/flow/ket_partner/exporter.py tests/ket_partner/test_db.py
git commit -m "refactor(db): add user_id multi-tenant support and RecentSentencesRepo"
```

---

### Task 2: Agent Refactoring (Stateless Agent & Config-Driven State)

**Files:**
- Modify: `src/flow/ket_partner/agent.py`
- Modify: `src/flow/ket_partner/main.py`
- Modify: `tests/ket_partner/conftest.py`
- Modify: `tests/ket_partner/test_graph_integration.py`

**Interfaces:**
- Consumes: Refactored `Repos.for_user(db, user_id)` from Task 1
- Produces: Stateless `KETPartnerAgent.__init__(self, llm_flash, llm_smart, config)`, `build_agent(llm_flash, llm_smart, db, checkpointer=None)`

- [ ] **Step 1: Write failing test for stateless KETPartnerAgent**

Add a test in `tests/ket_partner/test_graph_integration.py` testing stateless invocation with `RunnableConfig`:

```python
@pytest.mark.asyncio
async def test_stateless_agent_invocation(test_db):
    repos = Repos.for_user(test_db, "test_user")
    cfg = load_config()
    agent_instance = KETPartnerAgent(llm_flash, llm_smart, cfg)
    builder = StateGraph(BTPKetState)
    graph = await agent_instance.compile(builder)
    
    config = {
        "configurable": {
            "thread_id": "test_user:main",
            "user_id": "test_user",
            "repos": repos,
            "user_info": {"nickname": "小明", "age": 9},
        }
    }
    res = await graph.ainvoke({"messages": [HumanMessage(content="Hello")]}, config=config)
    assert "messages" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk pytest tests/ket_partner/test_graph_integration.py -k test_stateless_agent_invocation`
Expected: FAIL due to legacy `__init__` requiring `repos` and `info`.

- [ ] **Step 3: Refactor KETPartnerAgent to be stateless**

Modify `src/flow/ket_partner/agent.py`:
1. Update `__init__(self, llm_flash, llm_smart, config)`: remove `self.repos`, `self.info`, `self._recent_sentences`, `self._recent_scaffolding`.
2. Update all graph node signatures from `(self, state)` to `(self, state, config: RunnableConfig)`.
3. Inside nodes, extract `repos = config["configurable"]["repos"]` and `user_info = config["configurable"]["user_info"]`.
4. Update `init_state` to perform message windowing (keeping latest 10 messages).
5. Update `build_agent` signature: `async def build_agent(llm_flash, llm_smart, db, checkpointer=None)`.
6. Update `src/flow/ket_partner/main.py` and `tests/ket_partner/conftest.py` to match the new signature.

- [ ] **Step 4: Run full agent test suite to verify it passes**

Run: `rtk pytest tests/ket_partner/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/flow/ket_partner/agent.py src/flow/ket_partner/main.py tests/ket_partner/
git commit -m "refactor(agent): make KETPartnerAgent stateless and config-driven"
```

---

### Task 3: FastAPI Backend & API Endpoints

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/settings.py`
- Create: `src/api/schemas.py`
- Create: `src/api/deps.py`
- Create: `src/api/routes/__init__.py`
- Create: `src/api/routes/chat.py`
- Create: `src/api/routes/report.py`
- Create: `src/api/routes/messages.py`
- Create: `src/api/app.py`
- Create: `src/api/main.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_chat.py`
- Create: `tests/api/test_report.py`
- Create: `tests/api/test_messages.py`

**Interfaces:**
- Consumes: `init_db()`, `build_agent()`, `Repos.for_user()`
- Produces: FastAPI App running on port 8000 serving `/api/chat`, `/api/report`, `/api/report/{category}`, `/api/messages` and static files.

- [ ] **Step 1: Write failing API integration tests using `httpx.AsyncClient`**

Create `tests/api/test_chat.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.app import app

@pytest.mark.asyncio
async def test_get_report_counts():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/report")
    assert response.status_code == 200
    data = response.json()
    assert "mastered_count" in data
    assert "total_words" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk pytest tests/api/test_chat.py`
Expected: FAIL (module `src.api.app` not found).

- [ ] **Step 3: Implement `settings.py`, `schemas.py`, `deps.py`, routes and `app.py`**

1. `src/api/settings.py`: Define `Settings(BaseSettings)`.
2. `src/api/schemas.py`: Define Pydantic models `ChatRequest`, `ChatResponse`, `ReportResponse`, `ReportWord`, `ReportCategoryResponse`, `MessageOut`.
3. `src/api/deps.py`: Define `get_current_user`, `get_db`, `get_agent`, `get_settings`.
4. `src/api/routes/chat.py`: Implement `POST /api/chat`.
5. `src/api/routes/report.py`: Implement `GET /api/report` and `GET /api/report/{category}`.
6. `src/api/routes/messages.py`: Implement `GET /api/messages`.
7. `src/api/app.py`: Configure FastAPI with `lifespan`, exception handlers, routers, and StaticFiles mount for `web/dist`.
8. `src/api/main.py`: Entrypoint running `uvicorn.run("src.api.app:app", host=settings.HOST, port=settings.PORT)`.

- [ ] **Step 4: Run API tests to verify pass**

Run: `rtk pytest tests/api/ -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/api/ tests/api/
git commit -m "feat(api): implement FastAPI server and REST endpoints"
```

---

### Task 4: Vue 3 Frontend Scaffolding, API Client & Pinia Stores

**Files:**
- Create: `web/package.json`
- Create: `web/vite.config.ts`
- Create: `web/tsconfig.json`
- Create: `web/index.html`
- Create: `web/src/main.ts`
- Create: `web/src/App.vue`
- Create: `web/src/router.ts`
- Create: `web/src/api/client.ts`
- Create: `web/src/api/types.ts`
- Create: `web/src/stores/chat.ts`
- Create: `web/src/stores/report.ts`

**Interfaces:**
- Consumes: Backend REST API contracts from Task 3
- Produces: Vue 3 application shell, `client.ts` wrapper with relative path `/api`, `useChatStore` and `useReportStore`.

- [ ] **Step 1: Create `web/package.json`, `vite.config.ts`, `tsconfig.json`**

`web/package.json`:
```json
{
  "name": "ket-partner-web",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "pinia": "^2.1.7",
    "vue": "^3.4.21",
    "vue-router": "^4.3.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.0.4",
    "typescript": "^5.2.2",
    "vite": "^5.1.6",
    "vue-tsc": "^2.0.6"
  }
}
```

`web/vite.config.ts`:
```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: { '/api': 'http://localhost:8000' }
  }
})
```

- [ ] **Step 2: Implement `client.ts`, `types.ts`, `chat.ts`, and `report.ts`**

`web/src/api/client.ts`:
```typescript
const BASE = ''

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    let detail = text
    try {
      const json = JSON.parse(text)
      detail = json.detail || text
    } catch {}
    throw new Error(`API ${res.status}: ${detail}`)
  }
  return res.json()
}
```

Implement `types.ts`, `useChatStore` in `stores/chat.ts`, and `useReportStore` in `stores/report.ts` exactly as specified in the Spec.

- [ ] **Step 3: Verify TypeScript compilation**

Run in terminal: `cd web && npm install && npm run build` (or verify TS types).
Expected: Clean build output in `web/dist`.

- [ ] **Step 4: Commit changes**

```bash
git add web/
git commit -m "feat(web): scaffold Vue 3 project with Pinia stores and API client"
```

---

### Task 5: Vue 3 Views & Components Implementation

**Files:**
- Create: `web/src/components/WordListModal.vue`
- Create: `web/src/views/ChatView.vue`
- Create: `web/src/views/ReportView.vue`
- Modify: `web/src/App.vue`
- Modify: `web/src/router.ts`

**Interfaces:**
- Consumes: `useChatStore` and `useReportStore` from Task 4
- Produces: Complete UI for Chat and Report views with modal pagination.

- [ ] **Step 1: Implement `WordListModal.vue`**

Create `web/src/components/WordListModal.vue` with `<Teleport to="body">`, displaying category title, word list cards (word, context, mastery, exposed, correct, wrong, status), and pagination buttons (`prevPage` / `nextPage`).

- [ ] **Step 2: Implement `ChatView.vue`**

Create `web/src/views/ChatView.vue`:
- On mount: call `chatStore.load()` (`GET /api/messages?limit=15`).
- Message list container with auto-scroll to bottom.
- Bottom input form: text input + send button (disabled when `sending` is true).
- Optimistic user message append on submit.
- Error banner above input box with retry button if request fails.

- [ ] **Step 3: Implement `ReportView.vue` and `App.vue` top navigation**

Create `web/src/views/ReportView.vue`:
- On mount: call `reportStore.loadCounts()`.
- Render 5 clickable summary count cards (Mastered, Learning, Struggling, Used, Unused) with percentage calculations against `total_words`.
- Clicking card calls `reportStore.openCategory(cat)`.
- Render `<WordListModal>` when `activeCategory` is non-null.

Update `web/src/App.vue`:
- Top navigation bar with router links for `[对话]` (`/chat`) and `[报告]` (`/report`).
- `<RouterView />` container.

- [ ] **Step 4: Build frontend bundle and test static serving**

Run: `cd web && npm run build`
Expected: Success with files generated in `web/dist/`.

- [ ] **Step 5: Commit changes**

```bash
git add web/src/
git commit -m "feat(web): implement ChatView, ReportView, and WordListModal"
```

---

### Task 6: End-to-End Integration Verification & Code Standards Check

**Files:**
- Audit & Verify: All project files

**Interfaces:**
- Consumes: Complete backend and frontend codebase
- Produces: Verified working application in development and production modes with zero static analysis issues.

- [ ] **Step 1: Execute static analysis checks (Ruff, Mypy, Pytest)**

Run:
```bash
rtk ruff check src/ tests/
rtk mypy src/
rtk pytest tests/ -v
```
Expected: All 3 static checks pass with 0 errors / 0 warnings.

- [ ] **Step 2: Verify production build static file serving**

1. Build frontend: `cd web && npm run build`
2. Start server in production mode:
```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m uvicorn src.api.main:app --port 8000
```
3. Test GET `/` and GET `/api/report` using `httpx` or curl:
```bash
rtk curl http://localhost:8000/
rtk curl http://localhost:8000/api/report
```
Expected: HTTP 200 returned for both root index.html and API report endpoints.

- [ ] **Step 3: Final Commit**

```bash
git add .
git commit -m "chore(build): complete KET Partner Web App integration and verification"
```
