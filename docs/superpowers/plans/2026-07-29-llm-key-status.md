# LLM Key Status Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Web 前端能感知 DashScope API key 的可用性状态，在 Navbar 上实时反映指示灯并展开 Popover，并在 key 不可用时禁发消息与展示告警。

**Architecture:** 后端维持只读 module-level key 快照及内存可变 `LlmKeyStatus` 容器（单一写入者为 `chat.py` 路由）；暴露 `GET /api/llm/status` 端点；前端 Pinia `llmKeyStore` 首次异步拉取并随时拉取状态，`LlmStatusBadge.vue` 和 `ChatView.vue` 依据此状态拦截输入及展现告警。

**Tech Stack:** FastAPI, PyYAML, pytest, Vue 3, Pinia, TypeScript, Vitest, @vue/test-utils, jsdom.

## Global Constraints

- **Python Version & Tools**: Python 3.10+, FastAPI, PyYAML, pytest, ruff, mypy --strict.
- **Frontend Tools**: Vue 3, Pinia, TypeScript (vue-tsc), Vitest, @vue/test-utils, jsdom.
- **Error Tuple Rule**: 遵循 CLAUDE.md §一.1 & §一.5，严禁裸 except Exception，严格区分外部异常与代码 bug。
- **Single-Writer Rule**: `LlmKeyStatus` 的可变状态必须明确单一写入者并包含符合 §三.3 的 docstring 格式。
- **Non-blocking App Shell**: `main.ts` 同步挂载 Vue 应用，`loadStatus()` 异步执行，不阻塞 UI 渲染。

---

### Task 1: Backend LlmKeyStatus & Mask Helper

**Files:**
- Create: `src/api/llm_key.py`
- Test: `tests/api/test_mask_key.py`
- Test: `tests/api/test_llm_key_status.py`

**Interfaces:**
- Produces: `LlmKeyStatus` dataclass (`set_error`, `clear_error`, `state`), `mask_key(key: str) -> str | None`, `_read_current_key() -> str`

- [ ] **Step 1: Write failing tests for mask_key**

Create `tests/api/test_mask_key.py`:

```python
import pytest
from src.api.llm_key import mask_key


def test_mask_key_normal() -> None:
    assert mask_key("sk-abcdefghijklmno") == "sk-a***lmno"


def test_mask_key_short() -> None:
    assert mask_key("abc") == "***bc"


def test_mask_key_empty() -> None:
    assert mask_key("") is None


def test_mask_key_whitespace() -> None:
    assert mask_key("   ") is None


def test_mask_key_boundary_8() -> None:
    assert mask_key("abcdefgh") == "abcd***efgh"


def test_mask_key_boundary_7() -> None:
    assert mask_key("abcdefg") == "***fg"


def test_mask_key_strips_whitespace() -> None:
    assert mask_key("  sk-abcdefghijklmno  ") == "sk-a***lmno"
```

- [ ] **Step 2: Write failing tests for LlmKeyStatus**

Create `tests/api/test_llm_key_status.py`:

```python
import time
from typing import Generator
import pytest
from src.api.llm_key import LlmKeyStatus


def test_state_green_when_key_present_and_no_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "sk-xxx")
    status = LlmKeyStatus()
    assert status.state == "green"


def test_state_red_when_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "")
    status = LlmKeyStatus()
    assert status.state == "red"


def test_state_red_when_error_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "sk-xxx")
    status = LlmKeyStatus()
    status.set_error("auth error")
    assert status.state == "red"


def test_state_green_after_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "sk-xxx")
    status = LlmKeyStatus()
    status.set_error("auth error")
    status.clear_error()
    assert status.state == "green"


def test_state_ignores_whitespace_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "   ")
    status = LlmKeyStatus()
    assert status.state == "red"


def test_set_error_with_newer_timestamp_overwrites() -> None:
    status = LlmKeyStatus()
    status.set_error("old error", timestamp=100.0)
    status.set_error("new error", timestamp=200.0)
    assert status.last_error == "new error"
    assert status.last_error_updated_at == 200.0


def test_set_error_with_older_timestamp_does_not_overwrite() -> None:
    status = LlmKeyStatus()
    status.set_error("new error", timestamp=200.0)
    status.set_error("old error", timestamp=100.0)
    assert status.last_error == "new error"
    assert status.last_error_updated_at == 200.0


def test_clear_error_with_older_timestamp_does_not_clear() -> None:
    status = LlmKeyStatus()
    status.set_error("error", timestamp=200.0)
    status.clear_error(timestamp=100.0)
    assert status.last_error == "error"
    assert status.last_error_updated_at == 200.0


def test_clear_error_with_newer_timestamp_clears() -> None:
    status = LlmKeyStatus()
    status.set_error("error", timestamp=100.0)
    status.clear_error(timestamp=200.0)
    assert status.last_error is None
    assert status.last_error_updated_at == 200.0
```

- [ ] **Step 3: Run pytest to verify tests fail**

Run: `pytest tests/api/test_mask_key.py tests/api/test_llm_key_status.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api.llm_key'`

- [ ] **Step 4: Implement src/api/llm_key.py**

Create `src/api/llm_key.py`:

```python
import time
from dataclasses import dataclass
from typing import Literal

from flow.common import dashscope_api_key


@dataclass
class LlmKeyStatus:
    """LLM 可用性状态容器。

    共享可变状态,单一写入者契约(CLAUDE.md §三.2、§三.3):

    - last_error & last_error_updated_at: 仅 routes/chat.py 在 chat 鉴权失败或成功时写,
      记录状态变更时间戳以解决并发/交错竞态问题。其他位置只读。

    并发写入语义: guard 用写入时间(completion time, time.time())比较,即"最后完成的那次 chat 胜出",
    确保成功产出 AI 回复的请求能及时清空错误标记,真实反映系统的最新可用状态。
    """
    last_error: str | None = None
    last_error_updated_at: float | None = None

    def set_error(self, error: str, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        if self.last_error_updated_at is None or ts >= self.last_error_updated_at:
            self.last_error = error
            self.last_error_updated_at = ts

    def clear_error(self, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        if self.last_error_updated_at is None or ts >= self.last_error_updated_at:
            self.last_error = None
            self.last_error_updated_at = ts

    @property
    def state(self) -> Literal["red", "green"]:
        if not _read_current_key():
            return "red"
        if self.last_error is not None:
            return "red"
        return "green"


def _read_current_key() -> str:
    """从 src/flow/common.py 读取已解析的 dashscope key(已 strip)。"""
    return dashscope_api_key.strip()


def mask_key(key: str) -> str | None:
    """格式化 key 为掩码形式:
    - 空 / 纯空白 → None
    - 长度 < 8   → "***XX"(末 2 位)
    - 长度 ≥ 8   → "XXXX***XXXX"(前 4 + 后 4)
    """
    if not key or not key.strip():
        return None
    k = key.strip()
    if len(k) < 8:
        return f"***{k[-2:]}"
    return f"{k[:4]}***{k[-4:]}"
```

- [ ] **Step 5: Run pytest & static checks to verify tests pass**

Run: `pytest tests/api/test_mask_key.py tests/api/test_llm_key_status.py`
Expected: PASS

Run: `ruff check src/api/llm_key.py && mypy --strict src/api/llm_key.py`
Expected: Clean with no errors

- [ ] **Step 6: Commit Task 1**

```bash
git add src/api/llm_key.py tests/api/test_mask_key.py tests/api/test_llm_key_status.py
git commit -m "feat(backend): add LlmKeyStatus container and mask_key helper"
```

---

### Task 2: Backend Schemas, Route & App Lifespan Integration

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `src/api/deps.py`
- Modify: `src/api/app.py`
- Create: `src/api/routes/llm.py`
- Test: `tests/api/routes/test_llm_status.py`

**Interfaces:**
- Consumes: `LlmKeyStatus`, `_read_current_key`, `mask_key`
- Produces: `LlmStatusResponse`, `get_llm_key_status` dependency, `GET /api/llm/status` endpoint

- [ ] **Step 1: Write failing test for GET /api/llm/status**

Create `tests/api/routes/test_llm_status.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import app
from src.api.llm_key import LlmKeyStatus


@pytest.mark.asyncio
async def test_status_green_initial(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    db_file = str(tmp_path / "test_llm_status.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr("src.api.routes.llm._read_current_key", lambda: "sk-abcdefghijklmno")

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.get("/api/llm/status")
        assert res.status_code == 200
        data = res.json()
        assert data == {
            "state": "green",
            "masked_key": "sk-a***lmno",
            "last_error": None,
        }


@pytest.mark.asyncio
async def test_status_red_when_no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    db_file = str(tmp_path / "test_llm_status_nokey.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr("src.api.routes.llm._read_current_key", lambda: "")

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.get("/api/llm/status")
        assert res.status_code == 200
        data = res.json()
        assert data["state"] == "red"
        assert data["masked_key"] is None


@pytest.mark.asyncio
async def test_status_red_with_error(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    db_file = str(tmp_path / "test_llm_status_err.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr("src.api.routes.llm._read_current_key", lambda: "sk-abcdefghijklmno")

    async with app.router.lifespan_context(app):
        app.state.llm_key_status.set_error("auth error")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            res = await ac.get("/api/llm/status")
            assert res.status_code == 200
            data = res.json()
            assert data["state"] == "red"
            assert data["last_error"] == "auth error"
```

- [ ] **Step 2: Add LlmStatusResponse to src/api/schemas.py**

Append to `src/api/schemas.py`:

```python
class LlmStatusResponse(BaseModel):
    state: Literal["red", "green"]
    masked_key: str | None
    last_error: str | None
```

- [ ] **Step 3: Add get_llm_key_status to src/api/deps.py**

Add to `src/api/deps.py`:

```python
from typing import cast
from src.api.llm_key import LlmKeyStatus


async def get_llm_key_status(request: Request) -> LlmKeyStatus:
    return cast(LlmKeyStatus, request.app.state.llm_key_status)
```

- [ ] **Step 4: Create src/api/routes/llm.py**

Create `src/api/routes/llm.py`:

```python
from fastapi import APIRouter, Depends
from src.api.deps import get_llm_key_status
from src.api.llm_key import LlmKeyStatus, _read_current_key, mask_key
from src.api.schemas import LlmStatusResponse

router = APIRouter()


@router.get("", response_model=LlmStatusResponse)
async def get_status(
    status: LlmKeyStatus = Depends(get_llm_key_status),
) -> LlmStatusResponse:
    return LlmStatusResponse(
        state=status.state,
        masked_key=mask_key(_read_current_key()),
        last_error=status.last_error,
    )
```

- [ ] **Step 5: Register llm_key_status & router in src/api/app.py**

In `src/api/app.py`:
1. Import `LlmKeyStatus` and `llm` router module.
2. Initialize `app.state.llm_key_status = LlmKeyStatus()` inside `lifespan`.
3. Include router: `app.include_router(llm.router, prefix="/api/llm", tags=["llm"])`.

- [ ] **Step 6: Run tests & static checks**

Run: `pytest tests/api/routes/test_llm_status.py`
Expected: PASS

Run: `ruff check src/api/ && mypy --strict src/api/`
Expected: Clean with no errors

- [ ] **Step 7: Commit Task 2**

```bash
git add src/api/schemas.py src/api/deps.py src/api/routes/llm.py src/api/app.py tests/api/routes/test_llm_status.py
git commit -m "feat(backend): implement GET /api/llm/status route and app lifespan integration"
```

---

### Task 3: Chat Route Key Guard, Exception Mapping & Pytest Config

**Files:**
- Modify: `src/api/routes/chat.py`
- Modify: `src/flow/common.py`
- Create: `pyproject.toml`
- Create: `tests/api/routes/conftest.py`
- Create: `tests/api/routes/test_chat_route.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_chat_real_llm.py`
- Delete: `tests/api/test_chat.py`

**Interfaces:**
- Consumes: `LlmKeyStatus`, `_read_current_key`, `get_llm_key_status`
- Produces: Chat route key status state machine transition (401 auth error -> red state, success -> green state)

- [ ] **Step 1: Create pyproject.toml**

Create `pyproject.toml` in project root:

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]
markers = [
    "integration: marks tests that hit real external services (deselect with '-m \"not integration\"')",
]
```

- [ ] **Step 2: Fix except in src/flow/common.py**

In `src/flow/common.py`, update `except Exception as e:` on line ~56 to:

```python
except (yaml.YAMLError, OSError) as e:
    logger.warning(f"Failed to load API key from {pet_config}: {e}", exc_info=True)
```

Make sure `import yaml` is present at the top.

- [ ] **Step 3: Create tests/api/routes/conftest.py**

Create `tests/api/routes/conftest.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock

from langgraph.graph.state import CompiledStateGraph
from src.api.app import app
from src.api.deps import get_agent
from src.api.llm_key import LlmKeyStatus


@pytest.fixture
def mock_agent() -> AsyncMock:
    agent = AsyncMock(spec=CompiledStateGraph)
    agent.ainvoke = AsyncMock(return_value={"messages": []})
    return agent


@pytest.fixture
def llm_key_status() -> LlmKeyStatus:
    return LlmKeyStatus()


@pytest.fixture
async def client(mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    db_file = str(tmp_path / "test_chat_route.db")
    monkeypatch.setenv("DB_PATH", db_file)

    app.dependency_overrides[get_agent] = lambda: mock_agent

    async with app.router.lifespan_context(app):
        app.state.llm_key_status = llm_key_status

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()
```

- [ ] **Step 4: Create failing tests/api/routes/test_chat_route.py**

Create `tests/api/routes/test_chat_route.py`:

```python
import asyncio
import openai
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock

from src.api.llm_key import LlmKeyStatus


@pytest.mark.asyncio
async def test_chat_returns_503_when_no_key(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "")
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 503
    assert mock_agent.ainvoke.await_count == 0
    assert llm_key_status.last_error is None


@pytest.mark.asyncio
async def test_chat_401_on_auth_error(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = openai.AuthenticationError(
        message="invalid key", response=AsyncMock(), body=None
    )
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 401
    assert llm_key_status.last_error == "API key 无效或无权限"
    assert llm_key_status.state == "red"


@pytest.mark.asyncio
async def test_chat_401_on_bad_request(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = openai.BadRequestError(
        message="bad format key", response=AsyncMock(), body=None
    )
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 401
    assert llm_key_status.last_error == "API key 无效或无权限"


@pytest.mark.asyncio
async def test_chat_clears_error_on_success(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    llm_key_status.set_error("previous error")
    mock_msg = AsyncMock()
    mock_msg.content = "AI response"
    mock_agent.ainvoke.return_value = {"messages": [mock_msg]}

    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 200
    assert llm_key_status.last_error is None
    assert llm_key_status.state == "green"
    mock_agent.ainvoke.assert_awaited()


@pytest.mark.asyncio
async def test_chat_504_on_timeout(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = asyncio.TimeoutError()
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 504
    assert llm_key_status.last_error is None


@pytest.mark.asyncio
async def test_chat_504_on_sdk_timeout(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = openai.APITimeoutError(request=AsyncMock())
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 504
    assert llm_key_status.last_error is None


@pytest.mark.asyncio
async def test_chat_500_on_code_bug(
    client: AsyncClient, mock_agent: AsyncMock, llm_key_status: LlmKeyStatus, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.routes.chat._read_current_key", lambda: "sk-test")
    mock_agent.ainvoke.side_effect = KeyError("bug_field")
    res = await client.post("/api/chat", json={"text": "Hello"})
    assert res.status_code == 500
    assert llm_key_status.last_error is None
```

- [ ] **Step 5: Create tests/integration/conftest.py & test_chat_real_llm.py**

Create `tests/integration/conftest.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import app


@pytest.fixture
async def client(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    db_file = str(tmp_path / "test_integration.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
```

Create `tests/integration/test_chat_real_llm.py`:

```python
import os
import pytest
from httpx import ASGITransport, AsyncClient
from src.api.app import app


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="requires DASHSCOPE_API_KEY"
)
async def test_chat_succeeds_with_real_key(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = str(tmp_path / "test_chat_real.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/api/chat", json={"text": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "ai_reply" in data
        assert "turn_id" in data
        assert app.state.llm_key_status.last_error is None
```

Delete redundant historical test `tests/api/test_chat.py`:

```bash
git rm tests/api/test_chat.py
```

- [ ] **Step 6: Update src/api/routes/chat.py**

In `src/api/routes/chat.py`:
1. Import `openai`, `LlmKeyStatus`, `_read_current_key`, `get_llm_key_status`.
2. Add `llm_key_status: LlmKeyStatus = Depends(get_llm_key_status)` to signature.
3. Check `if not _read_current_key(): raise HTTPException(503, ...)` at entry.
4. Wrap `agent.ainvoke` with try/except matching exception tuple rules:
   - `asyncio.TimeoutError` -> 504
   - `openai.APITimeoutError` -> 504
   - `(openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError)` -> `llm_key_status.set_error(LLM_AUTH_ERROR_MSG)` -> 401
   - `(openai.APIConnectionError, openai.RateLimitError)` -> 429 / 502
   - On success: `llm_key_status.clear_error()`

- [ ] **Step 7: Run pytest & static checks**

Run: `pytest tests/api/ tests/integration/`
Expected: PASS (integration test skipped if no API key in env)

Run: `ruff check src/api/ src/flow/common.py && mypy --strict src/api/ src/flow/common.py`
Expected: Clean with no errors

- [ ] **Step 8: Commit Task 3**

```bash
git add pyproject.toml src/flow/common.py src/api/routes/chat.py tests/api/routes/conftest.py tests/api/routes/test_chat_route.py tests/integration/conftest.py tests/integration/test_chat_real_llm.py
git commit -m "feat(backend): add key status guard and auth exception handling in chat route"
```

---

### Task 4: Frontend API Client & Pinia Stores

**Files:**
- Modify: `web/src/api/client.ts`
- Create: `web/src/stores/llmKey.ts`
- Modify: `web/src/stores/chat.ts`
- Test: `web/src/stores/__tests__/chat.spec.ts`

**Interfaces:**
- Consumes: `GET /api/llm/status`, `POST /api/chat`
- Produces: `ApiError`, `useLlmKeyStore` (`state`, `maskedKey`, `lastError`, `loaded`, `popoverOpen`, `loadStatus`, `openPopover`, `closePopover`), updated `useChatStore`

- [ ] **Step 1: Upgrade web/src/api/client.ts**

Update `web/src/api/client.ts`:

```typescript
const BASE = ''

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

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
      const rawDetail = json.detail ?? text
      detail = typeof rawDetail === 'string' ? rawDetail : JSON.stringify(rawDetail)
    } catch (e) {
      console.warn('Failed to parse error response JSON', e)
    }
    throw new ApiError(res.status, `API ${res.status}: ${detail}`)
  }
  return res.json()
}
```

- [ ] **Step 2: Create web/src/stores/llmKey.ts**

Create `web/src/stores/llmKey.ts`:

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../api/client'

interface LlmStatus {
  state: 'red' | 'green'
  masked_key: string | null
  last_error: string | null
}

export const useLlmKeyStore = defineStore('llmKey', () => {
  const state = ref<'red' | 'green'>('red')
  const maskedKey = ref<string | null>(null)
  const lastError = ref<string | null>(null)
  const loaded = ref(false)
  const popoverOpen = ref(false)

  async function loadStatus() {
    try {
      const res = await api<LlmStatus>('/api/llm/status')
      state.value = res.state
      maskedKey.value = res.masked_key
      lastError.value = res.last_error
    } catch (e) {
      console.warn('loadStatus failed, keeping current state', e)
    } finally {
      loaded.value = true
    }
  }

  function openPopover() { popoverOpen.value = true }
  function closePopover() { popoverOpen.value = false }

  return { state, maskedKey, lastError, loaded, popoverOpen, loadStatus, openPopover, closePopover }
})
```

- [ ] **Step 3: Update web/src/stores/chat.ts with error mapping & status refresh**

In `web/src/stores/chat.ts`:

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { ApiError, api } from '../api/client'
import { useLlmKeyStore } from './llmKey'

interface Message {
  role: 'user' | 'ai'
  content: string
  turn_id: number | null
  created_at: string
}

interface ChatResponse {
  ai_reply: string
  turn_id: number
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sending = ref(false)
  const error = ref<string | null>(null)

  async function load() {
    messages.value = await api<Message[]>('/api/messages?limit=15')
  }

  async function send(text: string) {
    sending.value = true
    error.value = null
    const llmKeyStore = useLlmKeyStore()

    const optimistic: Message = {
      role: 'user',
      content: text,
      turn_id: null,
      created_at: new Date().toISOString(),
    }
    messages.value.push(optimistic)
    try {
      const res = await api<ChatResponse>('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ text }),
      })
      optimistic.turn_id = res.turn_id
      messages.value.push({
        role: 'ai',
        content: res.ai_reply,
        turn_id: res.turn_id,
        created_at: new Date().toISOString(),
      })
    } catch (e: unknown) {
      error.value = mapChatError(e)
      messages.value.pop()
      throw e
    } finally {
      sending.value = false
      await llmKeyStore.loadStatus().catch((e) => console.warn('refresh llm status failed', e))
    }
  }

  return { messages, sending, error, load, send }
})

function mapChatError(e: unknown): string {
  if (e instanceof ApiError) {
    return HTTP_STATUS_ERROR_MAP[e.status] ?? '服务异常,请重新发送'
  }
  return e instanceof Error ? e.message : String(e)
}

const HTTP_STATUS_ERROR_MAP: Record<number, string> = {
  503: 'LLM 未配置,请联系管理员',
  401: 'API key 异常,详情见右上角状态',
  504: '请求超时,请重新发送',
  502: '网络异常,请稍后重新发送',
  429: '请求过于频繁,请稍后再试',
}
```

- [ ] **Step 4: Create web/src/stores/__tests__/chat.spec.ts**

Create `web/src/stores/__tests__/chat.spec.ts`:

```typescript
import { setActivePinia, createPinia } from 'pinia'
import { describe, beforeEach, it, expect, vi } from 'vitest'
import { useChatStore } from '../chat'
import * as clientModule from '../../api/client'

vi.mock('../../api/client', async () => {
  const actual = await vi.importActual<typeof import('../../api/client')>('../../api/client')
  return {
    ...actual,
    api: vi.fn(),
  }
})

describe('chat store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('maps 503 error correctly and pops optimistic message', async () => {
    const apiMock = vi.mocked(clientModule.api)
    apiMock.mockRejectedValueOnce(new clientModule.ApiError(503, 'LLM not configured'))
    apiMock.mockResolvedValueOnce({ state: 'red', masked_key: null, last_error: null })

    const chatStore = useChatStore()
    await expect(chatStore.send('hello')).rejects.toThrow()

    expect(chatStore.error).toBe('LLM 未配置,请联系管理员')
    expect(chatStore.messages.length).toBe(0)
  })

  it('maps 401 error correctly', async () => {
    const apiMock = vi.mocked(clientModule.api)
    apiMock.mockRejectedValueOnce(new clientModule.ApiError(401, 'LLM auth failed'))
    apiMock.mockResolvedValueOnce({ state: 'red', masked_key: 'sk-a***lmno', last_error: 'API key 无效或无权限' })

    const chatStore = useChatStore()
    await expect(chatStore.send('hello')).rejects.toThrow()

    expect(chatStore.error).toBe('API key 异常,详情见右上角状态')
  })

  it('refreshes llm status on finally', async () => {
    const apiMock = vi.mocked(clientModule.api)
    apiMock.mockResolvedValueOnce({ ai_reply: 'hi', turn_id: 1 })
    apiMock.mockResolvedValueOnce({ state: 'green', masked_key: 'sk-a***lmno', last_error: null })

    const chatStore = useChatStore()
    await chatStore.send('hello')

    expect(apiMock).toHaveBeenCalledWith('/api/llm/status')
  })
})
```

- [ ] **Step 5: Commit Task 4**

```bash
git add web/src/api/client.ts web/src/stores/llmKey.ts web/src/stores/chat.ts web/src/stores/__tests__/chat.spec.ts
git commit -m "feat(frontend): implement ApiError, llmKey store, and updated chat store with error mapping"
```

---

### Task 5: Frontend UI Components, Views & Test Setup

**Files:**
- Modify: `web/package.json`
- Modify: `web/vite.config.ts`
- Create: `web/src/components/LlmStatusBadge.vue`
- Create: `web/src/components/__tests__/LlmStatusBadge.spec.ts`
- Modify: `web/src/App.vue`
- Modify: `web/src/views/ChatView.vue`
- Modify: `web/src/main.ts`

**Interfaces:**
- Consumes: `useLlmKeyStore`, `useChatStore`
- Produces: Rendered `LlmStatusBadge`, disabled state in `ChatView`, warnings banner, test suite runnable with `npm run test`

- [ ] **Step 1: Add Vitest devDependencies & scripts to web/package.json**

In `web/package.json`:
1. Add devDependencies:
   ```json
   "@vue/test-utils": "^2.4.5",
   "jsdom": "^24.0.0",
   "vitest": "^1.4.0"
   ```
2. Add scripts:
   ```json
   "test": "vitest run",
   "test:watch": "vitest"
   ```

- [ ] **Step 2: Update web/vite.config.ts**

Update `web/vite.config.ts`:

```typescript
/// <reference types="vitest" />
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
```

- [ ] **Step 3: Create web/src/components/LlmStatusBadge.vue**

Create `web/src/components/LlmStatusBadge.vue`:

```vue
<template>
  <div class="llm-status-badge" ref="badgeRef" @click.stop="togglePopover">
    <span class="dot" :class="llmKeyStore.state"></span>
    <span class="label">{{ llmKeyStore.state === 'green' ? 'LLM 可用' : 'LLM 不可用' }}</span>

    <div v-if="llmKeyStore.popoverOpen" class="popover">
      <div class="popover-row">
        <span class="dot" :class="llmKeyStore.state"></span>
        <span>{{ llmKeyStore.state === 'green' ? 'LLM 可用' : 'LLM 不可用' }}</span>
      </div>
      <div class="popover-row">
        <span class="row-label">当前 key:</span>
        <code>{{ llmKeyStore.maskedKey ?? '未配置' }}</code>
      </div>
      <div v-if="llmKeyStore.lastError" class="popover-row error-row">
        <span class="row-label">错误原因:</span>
        <span>{{ llmKeyStore.lastError }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from 'vue'
import { useLlmKeyStore } from '../stores/llmKey'

const llmKeyStore = useLlmKeyStore()
const badgeRef = ref<HTMLElement | null>(null)

function togglePopover() {
  llmKeyStore.popoverOpen ? llmKeyStore.closePopover() : llmKeyStore.openPopover()
}

function handleDocumentClick(e: MouseEvent) {
  if (llmKeyStore.popoverOpen && badgeRef.value && !badgeRef.value.contains(e.target as Node)) {
    llmKeyStore.closePopover()
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && llmKeyStore.popoverOpen) {
    llmKeyStore.closePopover()
  }
}

onMounted(() => {
  document.addEventListener('click', handleDocumentClick)
  document.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleDocumentClick)
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
.llm-status-badge {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.4rem 0.75rem;
  border-radius: 9px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: background 0.2s;
}
.llm-status-badge:hover {
  background: #f1f5f9;
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}
.dot.green { background: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2); }
.dot.red { background: #ef4444; box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2); }
.label { color: #475569; }

.popover {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  min-width: 240px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 0.85rem 1rem;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
  z-index: 200;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.popover-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
}
.popover-row code {
  background: #f1f5f9;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.8rem;
}
.row-label {
  color: #64748b;
  min-width: 70px;
}
.error-row {
  color: #991b1b;
}
</style>
```

- [ ] **Step 4: Create web/src/components/__tests__/LlmStatusBadge.spec.ts**

Create `web/src/components/__tests__/LlmStatusBadge.spec.ts`:

```typescript
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { describe, beforeEach, it, expect } from 'vitest'
import LlmStatusBadge from '../LlmStatusBadge.vue'
import { useLlmKeyStore } from '../../stores/llmKey'

describe('LlmStatusBadge.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders green state correctly', () => {
    const store = useLlmKeyStore()
    store.state = 'green'
    const wrapper = mount(LlmStatusBadge)
    expect(wrapper.find('.dot.green').exists()).toBe(true)
    expect(wrapper.text()).toContain('LLM 可用')
  })

  it('renders red state and opens popover on click', async () => {
    const store = useLlmKeyStore()
    store.state = 'red'
    store.maskedKey = 'sk-a***lmno'
    store.lastError = 'API key 无效'

    const wrapper = mount(LlmStatusBadge)
    expect(wrapper.find('.dot.red').exists()).toBe(true)
    expect(wrapper.find('.popover').exists()).toBe(false)

    await wrapper.find('.llm-status-badge').trigger('click')
    expect(store.popoverOpen).toBe(true)
    expect(wrapper.find('.popover').exists()).toBe(true)
    expect(wrapper.find('.popover').text()).toContain('API key 无效')
  })
})
```

- [ ] **Step 5: Update web/src/App.vue**

In `web/src/App.vue`:
1. Import `LlmStatusBadge` and `useLlmKeyStore`.
2. Add `<LlmStatusBadge v-if="llmKeyStore.loaded" />` in `.header-inner`.

- [ ] **Step 6: Update web/src/views/ChatView.vue**

In `web/src/views/ChatView.vue`:
1. Disable input & send button when `llmKeyStore.state === 'red'`.
2. Add red warning banner below chat header:
   ```html
   <div v-if="llmKeyStore.state === 'red'" class="llm-warning">
     <span>⚠️ LLM 不可用,请联系管理员配置 API key</span>
     <button class="warning-link" @click="llmKeyStore.openPopover()">查看详情</button>
   </div>
   ```
3. Remove retry button from error banner.
4. Implement `handleSubmit` with `try/catch` to restore `inputText` on error.

- [ ] **Step 7: Update web/src/main.ts**

In `web/src/main.ts`:
1. Synchronously mount app: `app.mount('#app')`.
2. Asynchronously load LLM status:
   ```typescript
   const llmKey = useLlmKeyStore(pinia)
   llmKey.loadStatus()
   ```

- [ ] **Step 8: Install dependencies & run frontend tests & build**

Run:
```bash
cd web && npm install && npm run test && npx vue-tsc --noEmit && npm run build
```
Expected: All frontend tests PASS, vue-tsc clean, build generates dist cleanly.

- [ ] **Step 9: Commit Task 5**

```bash
git add web/package.json web/vite.config.ts web/src/components/LlmStatusBadge.vue web/src/components/__tests__/LlmStatusBadge.spec.ts web/src/App.vue web/src/views/ChatView.vue web/src/main.ts
git commit -m "feat(frontend): add LlmStatusBadge UI, ChatView warning banner, and vitest testing setup"
```
