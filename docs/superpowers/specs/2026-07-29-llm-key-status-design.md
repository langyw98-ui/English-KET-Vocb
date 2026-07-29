# LLM Key Status Monitoring Design

**Date:** 2026-07-29
**Status:** Design (awaiting plan)
**Owner:** 狂暴棕熊 + Claude

## Goal

让 Web 前端能感知 DashScope API key 的可用性状态,并在 key 不可用时阻止用户发送消息,避免无效请求与困惑。

具体能力:

1. Navbar 上一个红/绿圆点控件,实时反映 LLM 可用性
2. 点击圆点展开 popover,显示当前状态、masked key、错误原因
3. key 为空或最近一次 chat 鉴权失败时,圆点为红;ChatView 输入框禁用,显示"请联系管理员"提示
4. chat 路径隐式驱动状态迁移(成功 → 绿,鉴权失败 → 红),无需独立的 validate 端点

## Non-goals(明确不做的事)

- **不做前端配置 UI**:API key 的配置完全是运维侧职责,通过 env var(`DASHSCOPE_API_KEY` / `QWEN_API_KEY` / `OPENAI_API_KEY`)或 `~/.config/pet/config.yaml` 完成。前端没有任何输入 key、保存 key 的入口。
- **不做 key 持久化**:不在浏览器 localStorage / sessionStorage / cookie 存 key;不在后端文件写 key。`flow.common.dashscope_api_key` 模块变量是 import 时一次性赋值的只读量。
- **不做热生效机制**:key 不会在运行时被替换,因此不需要 `make_llm_flash(key)` 工厂、不需要 `rebuild_lock`、不需要 agent 重建流程。
- **不做独立的 validate 端点**:chat 路径自己就是 validator,见下方"状态机"。
- **不做"忘记 key" / DELETE 端点**:运行期 key 不可清空,只能重启服务。
- **不做黄色三态**:状态二态(红/绿),不存在"未验证"中间态。
- **不做 chat 重试 UI**:错误条只展示消息,无重试按钮。用户失败后想重发需重新点发送(input 文字不回填)。
- **不做多 worker 状态共享**:当前 `main.py` 默认单进程;`uvicorn --workers N` 部署下各 worker 的 `LlmKeyStatus` 不共享,是已知限制。

## 与既有 spec 的关系

本设计**部分修订** `2026-07-08-web-app-design.md` 的 Non-goals:

- 旧 spec:"不做前端单元测试 / E2E:A 阶段手动测够"
- 本设计:**前端必须有单元测试**,且本次同步引入 vitest + @vue/test-utils + jsdom 测试基础设施(详见"前端测试基础设施"段)

其他部分(整体架构、FastAPI + Vue 3 选型、文件结构)与旧 spec 完全一致。

## Architecture

### 数据流

```
[Operator] -- writes --> [env var / ~/.config/pet/config.yaml]
                                  │
                                  ▼  (server startup, run once)
                   [src/flow/common.py: _resolve_dashscope_api_key()]
                                  │
                                  ▼  (read once, never written thereafter)
                          [dashscope_api_key (str, module-level)]
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
   [llm_flash]               [llm_max]              [LLMKeyStatus]
   (module singleton,        (module singleton,     (mutable validity flag,
    key bound at import)      key bound at import)   single field: last_error,
                                                    written only by chat path)

        │                                                 │
        ▼                                                 ▼
   [build_agent in lifespan]                    [GET /api/llm/status]
        │                                                 │
        ▼                                                 ▼
   [app.state.agent]                              [Frontend badge]
        │                                                 │
        ▼                                                 ▼
   [POST /api/chat]                              [popover: state +
        │                                         masked_key + error]
        ├── 503 if key empty (defense in depth)
        ├── on success: LlmKeyStatus.last_error = None (green)
        └── on auth fail: LlmKeyStatus.last_error = "..." (red)
```

### 模块边界(按 CLAUDE.md §十.1)

**后端:**

| 文件 | 职责 | 状态 |
|---|---|---|
| `src/api/llm_key.py` | `LlmKeyStatus` 容器、`mask_key()`、`_read_current_key()` | 新增 |
| `src/api/routes/llm.py` | `GET /api/llm/status` 路由 | 新增 |
| `src/api/routes/chat.py` | 入口 key 空检查 + 包裹 agent.ainvoke 捕获鉴权异常 + 状态迁移 | 改动 |
| `src/api/schemas.py` | 追加 `LlmStatusResponse` | 改动 |
| `src/api/app.py` | lifespan 初始化 `app.state.llm_key_status`;注册 llm 路由 | 改动 |
| `src/api/deps.py` | 追加 `get_llm_key_status` 依赖 | 改动 |
| `src/flow/common.py` | 修 `except Exception` 违规(§一.1) | 改动 |

**前端:**

| 文件 | 职责 | 状态 |
|---|---|---|
| `web/src/components/LlmStatusBadge.vue` | navbar 圆点 + 文字 + 点击展开 popover | 新增 |
| `web/src/stores/llmKey.ts` | Pinia store:`state` / `maskedKey` / `lastError` / `popoverOpen` + `loadStatus()` | 新增 |
| `web/src/api/client.ts` | 升级为 `ApiError`(带 `status` 字段) | 改动 |
| `web/src/App.vue` | navbar 挂载 `<LlmStatusBadge />` | 改动 |
| `web/src/views/ChatView.vue` | 红态禁用输入 + 警告条 + 错误文案精准映射 | 改动 |
| `web/src/stores/chat.ts` | `send()` 失败按状态码映射文案;finally 刷新 llmKey 状态 | 改动 |
| `web/src/main.ts` | 同步 mount + 异步 `llmKeyStore.loadStatus()`(不阻塞渲染) | 改动 |
| `web/vite.config.ts` | 合并 vitest 配置(`test` block + `/// <reference types="vitest" />`) | 改动 |
| `web/package.json` | devDependencies 加 `vitest` / `@vue/test-utils` / `jsdom`;scripts 加 `test` / `test:watch` | 改动 |

## Backend Design

### `src/api/llm_key.py`(新增,完整)

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
    """从 src/flow/common.py 读取已解析的 dashscope key(已 strip)。

    flow.common.dashscope_api_key 在模块 import 时一次性赋值(`str` 类型,空时为 ""),
    之后不变,天然只读(单一写入者 = import 时一次性赋值)。

    返回 strip 后的结果,使所有调用方(state 属性、chat 入口检查、mask_key)
    对空白 key 的判定一致:空字符串与纯空白字符串都视为"未配置"。

    注:所有业务与测试代码必须通过此函数读取 key,禁止直接 `from flow.common import dashscope_api_key`,
    防止 Python 模块级 import 导致的快照不可变问题。
    """
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

`_read_current_key` 用顶层 import 而非函数内 import。理由:`flow.common` 不 import `src.api`,无循环依赖;函数内 import 是过度防御,且每次 `state` 属性访问都走一次 `sys.modules` lookup,有不必要的开销。顶层 import 让"读取 import 时的 key 快照"这一意图更显式。业务与测试层统一调用 `_read_current_key()`。

### `src/api/schemas.py`(追加)

```python
from typing import Literal
from pydantic import BaseModel


class LlmStatusResponse(BaseModel):
    state: Literal["red", "green"]
    masked_key: str | None
    last_error: str | None
```

### `src/api/routes/llm.py`(新增)

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

纯读取,无副作用,无异常分支。

### `src/api/routes/chat.py`(改动)

**与 §十.3 的关系(有意取舍)**:chat 路由作为 openai SDK 异常 → HTTP 状态码的 adapter 层,直接 `import openai` + 捕具体异常类是 §十.3 的有意取舍。agent / llm client 内部不二次封装业务异常——若未来替换 LLM provider,需要同步调整此处的 except 元组。本取舍在 spec 中显式声明,避免未来 reviewer 拿 §十.3 来挑战。

```python
import asyncio
import logging
from typing import Final

import openai
from fastapi import HTTPException

from src.api.llm_key import LlmKeyStatus, _read_current_key

logger = logging.getLogger("ket_partner")

LLM_AUTH_ERROR_MSG: Final[str] = "API key 无效或无权限"


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
    agent: CompiledStateGraph = Depends(get_agent),
    settings: Settings = Depends(get_settings),
    llm_key_status: LlmKeyStatus = Depends(get_llm_key_status),
) -> ChatResponse:
    # 纵深防御:key 为空直接 503,不进 agent
    # _read_current_key() 已 strip,空白 key 也视为未配置
    if not _read_current_key():
        raise HTTPException(status_code=503, detail="LLM key not configured")

    req_start_time = time.time()
    repos = Repos.for_user(db, user.id)
    user_info = {"nickname": user.nickname, "age": user.age}

    try:
        result_state = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=req.text)]},
                config={
                    "configurable": {
                        "thread_id": f"{user.id}:main",
                        "user_id": user.id,
                        "repos": repos,
                        "user_info": user_info,
                    }
                },
            ),
            timeout=settings.REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError as e:
        # 外层 wait_for 超时:临时性问题,与 key 有效性无关,不污染状态。补全 warning 日志痕迹 (CLAUDE.md §一.3)
        logger.warning("agent execution timeout: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail="agent timeout")
    except openai.APITimeoutError as e:
        # SDK 内部超时(早于外层 wait_for 触发),同样视为超时映射到 504,保持文案一致。
        # 注意:必须在 APIConnectionError 之前捕获,因为 APITimeoutError 是其子类。
        logger.warning("LLM SDK timeout: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail="LLM timeout")
    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError) as e:
        # 鉴权 / 权限 / key 格式失败:key 状态 → 红。
        # BadRequestError 归入此元组(M6):DashScope 在 key 格式错误(缺前缀/含非法字符)时返回 400,
        # 与 auth fail 同属"key 配置问题",用户视角一致(都应联系管理员)。
        # timestamp 用完成写入时间 time.time():guard 语义为"最后完成胜出",真实反映系统可用性。
        llm_key_status.set_error(LLM_AUTH_ERROR_MSG)
        logger.warning("LLM auth/key failed: %s", e, exc_info=True)
        raise HTTPException(status_code=401, detail="LLM auth failed")
    except (openai.APIConnectionError, openai.RateLimitError) as e:
        # 外部临时失败:不污染状态,返回明确状态码
        logger.warning("LLM transient failure: %s", e, exc_info=True)
        status_code = 429 if isinstance(e, openai.RateLimitError) else 502
        raise HTTPException(status_code=status_code, detail=str(e) or "transient error")
    # 注意:不捕获 KeyError / ValueError / AttributeError 等代码 bug 类异常
    # (CLAUDE.md §一.5)。它们穿透到 app.py 全局 handler 返回 500,被测试捕获。

    # chat 成功:清掉 last_error,状态保持/转绿 (使用完成写入时间 time.time())
    llm_key_status.clear_error()

    messages = result_state.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="agent returned empty messages")

    # 跨边界字符串显式去空白非空校验 (CLAUDE.md §二.10)
    ai_text = str(messages[-1].content).strip()
    if not ai_text:
        raise HTTPException(status_code=500, detail="agent returned blank reply")

    profile = await repos.profile.get()
    turn_id = profile.get("total_turns", 0)

    return ChatResponse(ai_reply=ai_text, turn_id=turn_id)
```

### `src/api/app.py`(改动)

lifespan 增加 `app.state.llm_key_status` 初始化;路由表追加 llm router。

**与 §一.1 的关系(cleanup 例外声明)**:现有 lifespan finally 中的 `agent.aclose()` 与 `db.close()` 两处 `except Exception` **保留不变**。理由:cleanup finally 是 safety net 场景,目的是"清理失败不影响后续清理";若改成具体异常元组,langgraph / aiosqlite 的 close 异常类型未必能完整枚举,极端情况会漏捕导致 cleanup 中断。**这是 §一.1 的有意例外**,在 spec 中显式声明,本次不变更这两处。

```python
from src.api.llm_key import LlmKeyStatus
from src.api.routes import chat, llm, messages, report

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... 现有 db / agent 初始化 ...
    app.state.settings = settings
    app.state.db = db
    app.state.agent = agent
    app.state.llm_key_status = LlmKeyStatus()  # last_error=None → state 由 key 决定(非空则 green)
    yield
    # ... 现有 cleanup ...

# 路由注册
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(messages.router, prefix="/api/messages", tags=["messages"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])
```

### `src/api/deps.py`(追加)

> 现有 deps.py 顶部已 `from fastapi import HTTPException, Request`,本次只追加 cast 与新 dep。

```python
from typing import cast

from src.api.llm_key import LlmKeyStatus


async def get_llm_key_status(request: Request) -> LlmKeyStatus:
    # cast 让 mypy --strict 通过:app.state 是 Starlette.State,
    # 属性访问在 strict 下推断为 Any;cast 把运行时已经 lifespan 初始化好的实例显式标注类型。
    # 现有 get_agent / get_db / get_settings 同样有此模式,本次 spec 不变更。
    return cast(LlmKeyStatus, request.app.state.llm_key_status)
```

### `src/flow/common.py`(改动,仅一处)

```python
# 现状(src/flow/common.py:56)
except Exception as e:
    logger.warning(f"Failed to load API key from {pet_config}: {e}")

# 改为(只捕获具体外部失败类型,符合 CLAUDE.md §一.1、§一.3、§一.5)
# 注:不捕 UnicodeDecodeError——它是 ValueError 子类,§一.5 禁止捕获可能由代码 bug
# 引发的通用异常;若 yaml 文件实际编码与 encoding='utf-8' 不符,应作为配置问题暴露
# (修改 yaml 文件编码或修正 open 调用的 encoding 参数),而非被 except 吞掉。
except (yaml.YAMLError, OSError) as e:
    logger.warning(f"Failed to load API key from {pet_config}: {e}", exc_info=True)
```

需要 `import yaml`(若未 import)。`yaml` 来自 PyYAML,已是项目依赖。

## Frontend Design

### `web/src/api/client.ts`(升级)

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

`ApiError` 继承 `Error`,所有现有 `catch (e)` 调用方读 `e.message` 不破坏;新调用方可 `e instanceof ApiError` 拿 `.status`。

### `web/src/stores/llmKey.ts`(新增)

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
  // 默认 red(safe default:状态未知时按"不可用"处理,避免误导用户)。
  // loadStatus 完成后更新为真实状态;失败时维持 red(网络/服务异常时宁可显示不可用)。
  const state = ref<'red' | 'green'>('red')
  const maskedKey = ref<string | null>(null)
  const lastError = ref<string | null>(null)
  // loaded:首次 loadStatus 完成前为 false。
  // 用途:App.vue 用 v-if="loaded" 控制 LlmStatusBadge 渲染时机,
  // 避免 badge 在状态未知时显示误导性颜色(default red 会闪一下红 → 真实绿)。
  // ChatView 不 gated(始终渲染),但其输入框 :disabled 受 state 控制,
  // 默认 red → 初始禁用直到 loadStatus 完成,防止用户在状态未知时发请求。
  const loaded = ref(false)

  // popoverOpen 写入者枚举(§三.3 单一写入者契约的弱化形式 —— UI 状态多写入者):
  // - LlmStatusBadge.togglePopover (本组件点击切换)
  // - LlmStatusBadge.handleDocumentClick / handleKeydown (outside click / ESC 关闭)
  // - ChatView 警告条"查看详情"按钮 (openPopover,仅在 state=red 时由 v-if 守卫触发)
  const popoverOpen = ref(false)

  async function loadStatus() {
    // 网络/服务异常时不抛错——store 维持当前状态(默认 red 或上次成功值),
    // 避免 main.ts 同步 mount 后异步 loadStatus 失败导致视觉错乱。
    // 不静默吞错(§一.3):catch 留 console.warn 痕迹。
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

### `web/src/components/LlmStatusBadge.vue`(新增)

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

outside-click 用 document listener + `badgeRef.contains(target)` 判断,不引入 `@vueuse/core`。`@click.stop` 阻止 badge 点击冒泡到 document(否则点击 badge 自己也会触发 outside-click 立刻关闭)。

### `web/src/App.vue`(改动)

navbar `.header-inner` 里,`.nav-tabs` 之后追加:

```vue
<LlmStatusBadge v-if="llmKeyStore.loaded" />
```

`v-if="loaded"` 让 badge 在首次 `loadStatus` 完成后才渲染,避免在状态未知时显示误导性颜色(default red 会闪一下红 → 真实绿)。ChatView 不受此 gated(始终渲染,但输入框受 state 控制)。

```vue
<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import LlmStatusBadge from './components/LlmStatusBadge.vue'
import { useLlmKeyStore } from './stores/llmKey'

const llmKeyStore = useLlmKeyStore()
</script>
```

### `web/src/stores/chat.ts`(改动)

```typescript
import { ApiError, api } from '../api/client'
import { useLlmKeyStore } from './llmKey'

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
    const llmKeyStore = useLlmKeyStore()   // 函数内取,避免 store 初始化循环

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
      throw e // 抛出异常供调用方 (ChatView) 捕获并还原 inputText，防止用户输入丢失
    } finally {
      sending.value = false
      // 无论成败,刷新 LLM 状态(chat 成功可能清 last_error → 绿;鉴权失败 → 红)
      // 不静默吞错(§一.3):catch 留 console.warn 痕迹
      await llmKeyStore.loadStatus().catch((e) => console.warn('refresh llm status failed', e))
    }
  }

  return { messages, sending, error, load, send }
})

```typescript
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

### `web/src/views/ChatView.vue`(改动)

**1. 输入区禁用条件加红态判断:**

```vue
<input
  v-model="inputText"
  type="text"
  class="chat-input"
  placeholder="输入英语句子或对话内容..."
  :disabled="chatStore.sending || llmKeyStore.state === 'red'"
  ref="inputRef"
/>
<button
  type="submit"
  class="send-btn"
  :disabled="!inputText.trim() || chatStore.sending || llmKeyStore.state === 'red'"
>
  <span v-if="!chatStore.sending">发送 &rarr;</span>
  <span v-else class="btn-spinner"></span>
</button>
```

**2. 红态警告条(chat-header 下方、messages-list 上方):**

```vue
<div v-if="llmKeyStore.state === 'red'" class="llm-warning">
  <span>⚠️ LLM 不可用,请联系管理员配置 API key</span>
  <button class="warning-link" @click="llmKeyStore.openPopover()">查看详情</button>
</div>
```

```css
.llm-warning {
  background: #fef2f2;
  border-bottom: 1px solid #fca5a5;
  padding: 0.65rem 1.25rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #991b1b;
}
.warning-link {
  background: transparent;
  border: 1px solid #fca5a5;
  color: #991b1b;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  font-size: 0.8rem;
  cursor: pointer;
}
.warning-link:hover { background: #fee2e2; }
```

**3. 错误条去除重试按钮:**

```vue
<div v-if="chatStore.error" class="error-banner">
  <span class="error-icon">⚠️</span>
  <span class="error-text">{{ chatStore.error }}</span>
</div>
```

删除原来的 `<button class="retry-btn" @click="handleRetry">重试</button>`,以及对应的 `handleRetry`、`lastSentText`、`.retry-btn` 样式。

**4. script setup (增加 handleSubmit 异常捕获还原输入框文本):**

```typescript
import { ref, onMounted, nextTick, watch } from 'vue'
import { useChatStore } from '../stores/chat'
import { useLlmKeyStore } from '../stores/llmKey'

const chatStore = useChatStore()
const llmKeyStore = useLlmKeyStore()
const inputText = ref('')

async function handleSubmit() {
  const text = inputText.value.trim()
  if (!text || chatStore.sending || llmKeyStore.state === 'red') return
  inputText.value = ''
  try {
    await chatStore.send(text)
  } catch {
    // 捕获 send 抛出的异常，恢复用户输入的文本，防止丢失
    inputText.value = text
  }
}
```

### `web/src/main.ts`(改动)

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'
import { useLlmKeyStore } from './stores/llmKey'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(router)

// 同步挂载 app:ChatView 立即可见(输入框默认禁用, 因 store 默认 state=red),
// 避免 loadStatus 网络迟滞阻塞 App Shell 渲染(防白屏)。
app.mount('#app')

// 挂载后异步拉取 LLM 状态:
// - store.loaded 翻 true 后, LlmStatusBadge 才渲染(避免 navbar 显示误导性颜色)
// - store.state 更新后, ChatView 输入框按真实状态决定是否可用
// loadStatus 内部 try/catch 保护, 失败时维持 red(安全默认)
const llmKey = useLlmKeyStore(pinia)
llmKey.loadStatus()
```

同步挂载确保 ChatView 立即渲染;`LlmStatusBadge` 在 `loaded=true` 后渲染(避免 navbar 在状态未知时显示颜色);ChatView 输入框初始禁用直到 `loadStatus` 完成(防止用户在状态未知时发请求)。


### 前端测试基础设施(本次同步引入)

当前 `web/package.json` 无任何测试依赖。本次同步安装:

**`web/package.json` devDependencies 追加:**

```json
{
  "devDependencies": {
    "@vue/test-utils": "^2.4.5",
    "jsdom": "^24.0.0",
    "vitest": "^1.4.0"
  }
}
```

**`web/vite.config.ts` 合并测试配置:**

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

**`web/package.json` scripts 追加:**

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

### 前端测试用例

**`web/src/components/__tests__/LlmStatusBadge.spec.ts`:**

| 测试 | 断言 |
|---|---|
| 绿态渲染 | 找到 `.dot.green` + "LLM 可用" 文字 |
| 红态渲染 | 找到 `.dot.red` + "LLM 不可用" |
| 点击展开 popover | store `popoverOpen === true`、popover DOM 可见 |
| 再次点击关闭 | popover 消失 |
| outside click 关闭 | popover 打开后,document body 触发 click,popover 消失 |
| 点击 badge 自身不关闭 | popover 打开后,模拟 badge 元素自身 click,popover 保持打开(回归保护 `@click.stop` 修饰符,防止 outside-click handler 误吞 badge 内部点击) |
| ESC 关闭 | popover 打开后触发 keydown ESC,popover 消失 |
| maskedKey null 时显示"未配置" | popover 中找到"未配置" |
| lastError 非空时显示错误行 | popover 中找到错误原因文字 |

**`web/src/stores/__tests__/chat.spec.ts`:**

| 测试 | mock 设置 | 断言 |
|---|---|---|
| send 成功 | `api` 返回 ChatResponse | messages 多 1 条 AI 消息;`error` 为 null |
| send 503 | `api` 抛 `ApiError(503, ...)` | `error.value === 'LLM 未配置,请联系管理员'`;optimistic 消息被 pop |
| send 401 | `api` 抛 `ApiError(401, ...)` | `error.value === 'API key 异常,详情见右上角状态'` |
| send 504 | `api` 抛 `ApiError(504, ...)` | `error.value === '请求超时,请重新发送'` |
| send 后 loadStatus 被调用 | mock `../api/client` 的 `api` 函数(`vi.mock`),让 `/api/chat` 返回 ChatResponse、`/api/llm/status` 返回任意 LlmStatus | `api` 第 2 次调用的第 1 个参数 === `/api/llm/status`(成功路径与失败路径都要测,§六.4)。**不要 spy on `useLlmKeyStore`**——Pinia store 工厂函数的 spy 写法绕且脆弱,直接断言边界函数 `api` 的调用更可靠 |

## State Machine(形式化)

```
            ┌──────────────────────────────────────┐
            │         server startup               │
            │  _resolve_dashscope_api_key() → key  │
            └────────────────┬─────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       key 非空                         key 为空
       state = green                    state = red
       last_error = None                (永久红,直到重启)
              │
              │ chat 请求
              ▼
       ┌──────────────────┐
       │  agent.ainvoke   │
       └────────┬─────────┘
                │
       ┌────────┼────────────┬──────────────┐
       ▼        ▼            ▼              ▼
    成功     Timeout      AuthError    其他异常
       │     │            │              │
       │     │ 504        │ 401          │ 500
       │     │ 不改状态   │ last_error   │ 不改状态
       │     │            │ = "..."      │ (穿透)
       ▼     ▼            ▼              ▼
    last_error           state = red
    = None               (持续红,直到
    state = green         下次成功 chat)
    (保持/转绿)
```

**状态转移规则汇总:**

| 当前状态 | 触发事件 | 新状态 |
|---|---|---|
| (任何) | key 为空(读 `dashscope_api_key`) | red |
| (任何, key 非空) | chat 成功 | green |
| (任何, key 非空) | chat 鉴权失败(AuthenticationError / PermissionDeniedError / BadRequestError) | red |
| (任何, key 非空) | chat 超时 / 网络错误 / 限流 | 不变 |
| (任何, key 非空) | chat 其他异常(代码 bug) | 不变(异常穿透) |
| red(key 为空) | 重启服务 + 配 key | green |
| red(鉴权失败) | 重启服务 + 换有效 key | green(乐观,直到首次 chat) |

## Error Handling Matrix

`POST /api/chat` 异常分类与处理:

| 异常类型 | 来源 | HTTP | 更新 last_error? | 捕获? | 用户文案 |
|---|---|---|---|---|---|
| 入口 key 为空 | 服务端自检 | 503 | 否(本来就是红) | raise HTTPException | "LLM 未配置,请联系管理员" |
| `asyncio.TimeoutError` | 外层 wait_for 超时 | 504 | 否 | 是 | "请求超时,请重新发送" |
| `openai.APITimeoutError` | SDK 内部超时(`APIConnectionError` 子类,早于 wait_for 触发) | 504 | 否 | 是(须早于 `APIConnectionError` 捕获) | 同上 |
| `openai.AuthenticationError` | LLM 鉴权失败 | 401 | **是** | 是 + warning 日志 | "API key 异常,详情见右上角状态" |
| `openai.PermissionDeniedError` | LLM 权限不足 | 401 | **是** | 是 + warning 日志 | 同上 |
| `openai.BadRequestError` | key 格式错误等(DashScope 返回 400) | 401 | **是** | 是 + warning 日志 | 同上(M6:与 auth fail 同语义,都属"key 配置问题") |
| `openai.APIConnectionError` | 网络问题 | 502 | 否 | 是 | "网络异常,请稍后重新发送" |
| `openai.RateLimitError` | 限流 | 429 | 否 | 是 | "请求过于频繁,请稍后再试" |
| `KeyError`/`ValueError`/`AttributeError`/`IndexError` | 代码 bug | 500(全局) | 否 | **否**(穿透,§一.5) | "服务异常,请重新发送" |

**异常元组严格区分外部失败 vs 代码 bug**(CLAUDE.md §一.5):只捕获 openai SDK 的具体外部失败类型 + `asyncio.TimeoutError`;`ValueError` / `TypeError` / `KeyError` / `AttributeError` / `IndexError` 等代码 bug 类异常直接穿透,被全局 handler 返回 500 + 测试捕获。`BadRequestError` 归入鉴权失败元组(M6):虽然 SDK 上是 400,但 DashScope 在 key 格式错误时返回 400,与 auth fail 用户视角一致(都应联系管理员);若未来区分出"非 key 相关的 BadRequestError"(如 langchain 构造的请求体错误),需独立分类。

### 全局兜底:代码 bug 不会让进程崩溃

代码 bug 类异常穿透 `chat.py` 的 try/except 后,**不会**导致 uvicorn 进程退出。`src/api/app.py:88-91` 注册了 `Exception` 基类的全局 handler 作为 safety net:

```python
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception(f"unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "internal error"})
```

异常处理路径:

```
agent.ainvoke() 内部抛 KeyError
        │
        ▼  (chat.py 的 try/except 故意不捕获,§一.5)
propagate 出 chat() 函数
        │
        ▼
FastAPI 框架捕获
        │
        ▼
匹配到 @app.exception_handler(Exception)
        │
        ▼
logger.exception(...) 记完整 traceback
        │
        ▼
返回 JSONResponse(500, {"detail": "internal error"})
        │
        ▼
uvicorn event loop 继续运行,接受下一个请求
```

FastAPI + uvicorn 基于 async event loop,每个请求在独立 task 中执行。异常被框架捕获并转成 500 响应,**进程不退出,其他请求不受影响**。这是注册全局 handler 的核心目的:让任何意外异常都变成可控响应,而不是把服务搞挂。

**异常类型 vs 是否被全局 handler 捕获**:

| 异常 | 被捕获? | 行为 |
|---|---|---|
| `KeyError` / `ValueError` / `AttributeError` / `IndexError` / `TypeError` | ✅ 是(`Exception` 子类) | → 500 响应,进程继续 |
| `openai.*` SDK 异常 | ✅ 是(在 chat.py 被更具体的 except 提前捕获,不到这里) | → 对应状态码 |
| `asyncio.TimeoutError` | ✅ 是(在 chat.py 被捕获) | → 504 |
| `asyncio.CancelledError` | ❌ 否(`BaseException` 直接子类) | 任务取消信号,§七.3 要求只在 `finally` 清理,不捕获 |
| `SystemExit` / `KeyboardInterrupt` | ❌ 否(`BaseException` 直接子类) | 进程退出(预期的关机信号) |

**因此 `chat.py` 选择"不捕获代码 bug 异常"是安全的**:全局 handler 是兜底,services 不会因为单个请求里的字段名 typo 或类型错误而崩溃。测试 `test_chat_500_on_code_bug` 验证这条路径。

## Testing Strategy

按 CLAUDE.md §六 拆分。

### 后端单元测试

**`tests/api/test_mask_key.py`**(纯函数):

| 测试 | 输入 | 期望输出 |
|---|---|---|
| `test_mask_key_normal` | `"sk-abcdefghijklmno"` | `"sk-a***lmno"` |
| `test_mask_key_short` | `"abc"` | `"***bc"` |
| `test_mask_key_empty` | `""` | `None` |
| `test_mask_key_whitespace` | `"   "` | `None` |
| `test_mask_key_boundary_8` | `"abcdefgh"`(8 字符) | `"abcd***efgh"` |
| `test_mask_key_boundary_7` | `"abcdefg"`(7 字符) | `"***fg"` |
| `test_mask_key_strips_whitespace` | `"  sk-abcdefghijklmno  "` | `"sk-a***lmno"` |

**`tests/api/test_llm_key_status.py`**(状态容器 + timestamp guard):

| 测试 | 设置 | 断言 |
|---|---|---|
| `test_state_green_when_key_present_and_no_error` | monkeypatch `_read_current_key → "sk-xxx"`,`last_error=None` | `state == "green"` |
| `test_state_red_when_key_empty` | monkeypatch `_read_current_key → ""` | `state == "red"`(无视 last_error) |
| `test_state_red_when_error_set` | key 非空,`last_error="..."` | `state == "red"` |
| `test_state_green_after_clear_error` | 设过 last_error 后置 None | `state == "green"` |
| `test_state_ignores_whitespace_key` | monkeypatch `_read_current_key → "   "` | `state == "red"` |
| `test_set_error_with_newer_timestamp_overwrites` | 先 `set_error("old", timestamp=100)`,再 `set_error("new", timestamp=200)` | `last_error == "new"`;`last_error_updated_at == 200` |
| `test_set_error_with_older_timestamp_does_not_overwrite` | 先 `set_error("new", timestamp=200)`,再 `set_error("old", timestamp=100)` | `last_error == "new"`(未覆盖);`last_error_updated_at == 200` |
| `test_clear_error_with_older_timestamp_does_not_clear` | 先 `set_error("err", timestamp=200)`,再 `clear_error(timestamp=100)` | `last_error == "err"`(未清);`last_error_updated_at == 200` |
| `test_clear_error_with_newer_timestamp_clears` | 先 `set_error("err", timestamp=100)`,再 `clear_error(timestamp=200)` | `last_error is None`;`last_error_updated_at == 200` |
| `test_set_error_default_timestamp_uses_time_time` | monkeypatch `time.time` 返回固定值,`set_error("err")`(不传 timestamp) | `last_error_updated_at == <固定值>` |

monkeypatch `_read_current_key`(用 `monkeypatch.setattr("src.api.llm_key._read_current_key", lambda: "...")`),让测试不依赖 `src/flow/common.py` 的真实加载结果,hermetic。

timestamp guard 测试覆盖 spec 的核心并发防御机制(§六.1 兜底路径必须测)。

### 后端路由集成测试

**`tests/api/routes/test_chat_route.py`**(M4 改名,被测单元 = chat 路由):

| 测试 | mock | 断言 |
|---|---|---|
| `test_chat_returns_503_when_no_key` | monkeypatch `_read_current_key → ""`,`agent.ainvoke` AsyncMock | 响应 503;`agent.ainvoke.await_count == 0`(§六.4);`llm_key_status.last_error` 未变 |
| `test_chat_401_on_auth_error` | key 非空,`agent.ainvoke` side_effect=`openai.AuthenticationError(...)` | 响应 401;`llm_key_status.last_error == "API key 无效或无权限"` |
| `test_chat_401_on_permission_denied` | side_effect=`openai.PermissionDeniedError(...)` | 响应 401;`last_error` 被设 |
| `test_chat_401_on_bad_request` | side_effect=`openai.BadRequestError(...)`(key 格式错误) | 响应 401;`last_error` 被设(M6) |
| `test_chat_clears_error_on_success` | 预设 `last_error="..."`,`agent.ainvoke` 正常返回 | 响应 200;`last_error is None`;`agent.ainvoke.assert_awaited`(§六.4) |
| `test_chat_504_on_timeout` | side_effect=`asyncio.TimeoutError` | 响应 504;`last_error` 未变 |
| `test_chat_504_on_sdk_timeout` | side_effect=`openai.APITimeoutError(...)` | 响应 504;`last_error` 未变(回归保护:必须在 `APIConnectionError` 之前捕获) |
| `test_chat_502_on_connection_error` | side_effect=`openai.APIConnectionError(...)` | 响应 502;`last_error` 未变 |
| `test_chat_429_on_rate_limit` | side_effect=`openai.RateLimitError(...)` | 响应 429;`last_error` 未变 |
| `test_chat_500_on_code_bug` | side_effect=`KeyError("foo")` | 响应 500(全局 handler);`last_error` 未变;**异常类型穿透未被吞** |

**Mock 纪律**(§六.5 + 本次统一约定):
- **client fixture 形态**:见 `tests/api/routes/conftest.py`(方案 B:启动 lifespan + override agent + 覆盖 llm_key_status)。lifespan 启动真 DB / 真 Settings / 真 default user,只 mock agent 边界,符合 §六.5"mock 边界,不 mock 业务"。
- **agent 替换方式**:统一用 `app.dependency_overrides[get_agent] = lambda: mock_agent`(FastAPI 官方推荐),不要 `unittest.mock.patch("src.api.routes.chat.get_agent", ...)`。理由:dependency_overrides 让 mock 生命周期随 fixture,且不污染生产代码模块属性。
- `mock_agent` 用 `AsyncMock(spec=CompiledStateGraph)`,让 mypy --strict 通过;`agent.ainvoke = AsyncMock(return_value={"messages": [...]})`,每个测试自行 set return_value 或 side_effect。
- 断言用 `await_count` / `assert_awaited`(AsyncMock),**不能用 `MagicMock`**(无 await 支持,`assert_not_awaited` 不存在)
- `test_chat_returns_503_when_no_key` 必须断言 `await_count == 0`(§六.4)
- `test_chat_clears_error_on_success` 必须断言 `agent.ainvoke` 被 await 过(防止"直接走兜底返回也通过")

**`tests/api/routes/test_llm_status.py`:**

| 测试 | 设置 | 断言 |
|---|---|---|
| `test_status_green_initial` | key 非空,`last_error=None` | 响应 `{state: "green", masked_key: "sk-...", last_error: null}` |
| `test_status_red_when_no_key` | monkeypatch 空 key | 响应 `{state: "red", masked_key: null, last_error: null}` |
| `test_status_red_with_error` | 预设 `last_error="..."`,key 非空 | 响应 `{state: "red", masked_key: "sk-...", last_error: "..."}` |
| `test_status_mask_key_format` | key=`"sk-abcdefghijklmno"` | 响应 `masked_key == "sk-a***lmno"` |

### 真实 LLM 集成测试(§六.8)

**`tests/integration/test_chat_real_llm.py`**(M1 合并现有 `tests/api/test_chat.py` 的端到端断言 + 本次新增的 last_error 清除断言):

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
async def test_chat_succeeds_with_real_key(tmp_path, monkeypatch):
    # 用真实 env key 跑一次 chat;合并原 tests/api/test_chat.py 的端到端断言
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
        # 验证 chat 成功后 last_error 被清
        assert app.state.llm_key_status.last_error is None
```

`@pytest.mark.integration` + `skipif` 符合 §六.6 要求。无 key 环境(CI)自动 skip,不阻塞。

鉴权失败路径**不写真实测试**(无法稳定制造 401),靠 mock 覆盖。

### Conftest 形态与现有测试处置(B2 + B3)

**`tests/api/routes/conftest.py`(新增,方案 B:启动 lifespan + override agent + 覆盖 llm_key_status)**:

chat 路由 5 个 dep(get_agent / get_db / get_settings / get_current_user / get_llm_key_status)。其中 get_db / get_settings / get_current_user 直接或间接读 `app.state.db` / `app.state.settings`,而 `get_current_user` 在 `AUTH_MODE='disabled'` 下要执行真实 SQL 查 default user(`deps.py:31`)。方案 A(全 override)会让 `mock_db` 必须支持 `Repos.for_user` 内部所有 SQL chained 调用,极其脆弱。方案 B 启动 lifespan 让真 DB + 真 default user + 真 Settings 就位,只 override 边界(agent),符合 §六.5"mock 边界,不 mock 业务"。

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
    """CompiledStateGraph spec 让 mypy --strict 通过;ainvoke 由测试自行 set。"""
    agent = AsyncMock(spec=CompiledStateGraph)
    agent.ainvoke = AsyncMock(return_value={"messages": []})
    return agent


@pytest.fixture
def llm_key_status() -> LlmKeyStatus:
    """单一写入者契约下,每个测试拿独立实例避免相互污染。"""
    return LlmKeyStatus()


@pytest.fixture
async def client(mock_agent, llm_key_status, tmp_path, monkeypatch):
    """启动 lifespan(真 DB + 真 Settings + 真 default user)+ override agent + 覆盖 llm_key_status。

    - tmp_path + monkeypatch:每个测试用独立 SQLite 文件,hermetic(§六.6)
    - lifespan_context 启动:init_db 创建表 + 种子 default user;Settings 从 DB_PATH env 读取
    - dependency_overrides[get_agent]:替换为 mock_agent(被测单元的边界)
    - 覆盖 app.state.llm_key_status:让 fixture 提供的独立实例生效,避免测试间污染

    cleanup:dependency_overrides.clear() + lifespan 自动 teardown DB。
    """
    db_file = str(tmp_path / "test_chat_route.db")
    monkeypatch.setenv("DB_PATH", db_file)

    app.dependency_overrides[get_agent] = lambda: mock_agent

    async with app.router.lifespan_context(app):
        # lifespan 已 set app.state.llm_key_status = LlmKeyStatus(),
        # 立即覆盖为 fixture 提供的独立实例
        app.state.llm_key_status = llm_key_status

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()
```

**`tests/integration/conftest.py`(新增)**:沿用 lifespan_context 模式,但**不 override agent**(走真 agent + 真 DB)。`app_state` fixture 直接返回 `app.state`,让 integration 测试能访问 `app_state.llm_key_status`。

```python
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """启动真 lifespan(真 DB + 真 agent),不 override 任何 dep。"""
    db_file = str(tmp_path / "test_integration.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def app_state():
    """让测试能访问 app.state.llm_key_status 等运行时状态。"""
    return app.state
```

**现有 `tests/api/test_chat.py` 处置(B3,M1 方案 W)**:**删除,断言合并到 `tests/integration/test_chat_real_llm.py`**。

理由(§六.3 同一被测单元放同一文件 + §六.6 integration 必须标 + skipif):
- 现有 `tests/api/test_chat.py` 是真 LLM 端到端测试(走 lifespan + 真 DB + 真 agent + 真 `DASHSCOPE_API_KEY`),验 status_code=200 + 字段存在
- spec 新增 `tests/integration/test_chat_real_llm.py` 是真 LLM 测试,验 chat 成功后 `last_error` 被清
- 两者 purpose 高度重叠(都跑真 LLM 调用),保留两个 = CI 跑两次真 LLM,慢且消耗 key 配额
- 合并为单一测试同时覆盖两组断言;同时 `tests/api/test_chat.py` 缺 `@pytest.mark.integration` + skipif,违反 §六.6,合并后顺手修复

合并后形态详见下方"真实 LLM 集成测试"段落。

### pytest 配置(§六.7)

项目根目录当前**没有** `pyproject.toml` / `pytest.ini` / `setup.cfg`。本次同步新建 `pyproject.toml`(现代 Python 项目标准载体;未来 ruff/mypy 配置可追加到同一文件,本次只填 pytest 段):

**`pyproject.toml`**(新增,项目根目录):

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]
markers = [
    "integration: marks tests that hit real external services (deselect with '-m \"not integration\"')",
]
```

`filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]` 把"未注册 mark"作为错误,符合 §六.7。`pytest-asyncio` 配置沿用现有项目约定(已安装,现有 `@pytest.mark.asyncio` 测试能跑)。

ruff/mypy 配置**本次不追加**(独立 PR 处理);当前 Static Checks 命令行显式传文件路径,不读配置也能跑。

## Static Checks(§九)

每个 task 的 verify pass 必须包含三项全部清零:

```bash
# 后端:目录级检查(M7),所有改动文件全部覆盖
ruff check src/api/ src/flow/common.py
mypy --strict src/api/ src/flow/common.py
pytest tests/api/ tests/integration/
```

`pytest tests/integration/` 包含真 LLM 集成测试,无 `DASHSCOPE_API_KEY` 时自动 skip(§六.6 skipif),不阻塞 CI。

前端:

```bash
cd web && npm run test
cd web && npx vue-tsc --noEmit
cd web && npm run build
```

`npm run build` 验证 dist 产出正常。

## Acceptance Criteria

设计完成后,以下条件全部满足才算交付:

1. ✅ 启动时 `DASHSCOPE_API_KEY` 未设 → store 默认 red,ChatView 渲染但输入禁用,警告条显示"LLM 不可用,请联系管理员配置 API key";`loadStatus` 完成后 `LlmStatusBadge` 渲染并显红
2. ✅ 启动时 key 存在且有效 → `loadStatus` 完成后 badge 显绿,ChatView 输入可用,chat 正常工作
3. ✅ 启动时 key 存在但无效 → `loadStatus` 完成后 badge 显绿(后端 `last_error=None` 即乐观绿),首次 chat 鉴权失败后变红,popover 显示"API key 无效或无权限"
4. ✅ 点 badge 能展开 popover,显示状态 + masked_key(或"未配置")+ 错误原因(如有)
5. ✅ outside click 与 ESC 都能关闭 popover;点击 badge 自身不关闭(@click.stop 回归保护)
6. ✅ chat 错误条按状态码显示对应文案(503 / 401 / 504 / 502 / 429 / 500);`openai.APITimeoutError` 与 `asyncio.TimeoutError` 都映射到 504
7. ✅ chat 错误条无重试按钮
8. ✅ `src/flow/common.py:56` 的 `except Exception` 改为 `(yaml.YAMLError, OSError)`,**不含 `UnicodeDecodeError`**(让它穿透暴露配置问题)
9. ✅ `src/api/app.py` lifespan cleanup 中的两处 `except Exception` 保留不变(spec 显式声明为 §一.1 cleanup safety net 例外)
10. ✅ `loadStatus` 内部 try/catch,失败时 store 维持 red(安全默认);store `loaded` flag 控制 `LlmStatusBadge` 渲染时机(避免 navbar 显示误导性颜色);**main.ts 不会因 `/api/llm/status` 不可达而白屏**
11. ✅ `ruff check` / `mypy --strict` / `pytest` 全部清零
12. ✅ 前端 `npm run test` 通过(vitest 已安装,测试用例齐备);`npm run build` 验证 dist 产出正常
13. ✅ `openai.BadRequestError`(key 格式错误)归入鉴权失败元组,与 `AuthenticationError` / `PermissionDeniedError` 同样映射到 401 + red 状态
14. ✅ timestamp guard 测试覆盖:`set_error` / `clear_error` 在 older timestamp 下不覆盖、newer timestamp 下覆盖
15. ✅ 项目根目录新建 `pyproject.toml`,只含 `[tool.pytest.ini_options]` 段(filterwarnings + markers)

## Out of Scope(未来可扩展)

- **多 worker 部署的状态共享**:需要 redis 或共享存储;当前 `main.py` 默认单进程
- **单 worker 内多并发 chat 的 last_error 写入语义**:`last_error` 反映"最后**完成**的那次 chat"的结果(`set_error` / `clear_error` 采用写入时刻 `time.time()` 进行 timestamp 比较),确保成功产生 AI 回复的请求能及时清空错误标记,反向纠正历史错误。
- **启动时的乐观绿 (Optimistic Green) 取舍**:系统在启动且 key 非空时默认 `last_error=None`(显绿),直到首次 chat 触发鉴权失败变红。未引入启动独立 HTTP 校验探针是遵循 YAGNI 原则,视为已知行为。
- **AUTH_MODE="jwt" 多用户场景**:每个用户独立 key 状态,需要 session/cookie(本设计文档前面讨论过 D 方案,因当前不满足前置条件而排除)。**届时引入 JWT 时需同步区分 user auth 401 与 llm auth 401**(例如改用 419 或响应体加 `error_code` 字段),否则 `mapChatError` 的 `401 → "API key 异常"` 文案会与用户登录失效撞码
- **`src/flow/common.py` ChatOpenAI 重复构造代码的工厂抽取**:与本次需求无强绑定,独立 PR
- **轮询或 SSE 实时状态推送**:当前 init + chat 后刷新已足够
- **masked_key 之外的状态详情**(如最近验证时间、启动以来 chat 成功次数):本设计文档前面讨论过 C 选项,被排除

## File Map(实现时新增/改动的文件清单)

```
src/api/
├─ llm_key.py                    [新增]
├─ routes/llm.py                 [新增]
├─ routes/chat.py                [改动:加 key 检查 + 异常分类]
├─ schemas.py                    [改动:加 LlmStatusResponse]
├─ app.py                        [改动:lifespan + router]
└─ deps.py                       [改动:加 get_llm_key_status]

src/flow/common.py               [改动:except Exception 修复]

web/src/
├─ api/client.ts                 [改动:ApiError 类]
├─ stores/llmKey.ts              [新增]
├─ stores/chat.ts                [改动:mapChatError + loadStatus]
├─ components/LlmStatusBadge.vue [新增]
├─ components/__tests__/
│  └─ LlmStatusBadge.spec.ts     [新增]
├─ stores/__tests__/
│  └─ chat.spec.ts               [新增]
├─ views/ChatView.vue            [改动:禁用 + 警告条 + 去重试]
├─ App.vue                       [改动:挂载 badge + v-if loaded]
└─ main.ts                       [改动:同步 mount + 异步 loadStatus]

web/
├─ package.json                  [改动:devDeps + scripts]
└─ vite.config.ts                [改动:合并 vitest 配置]

tests/api/
├─ test_mask_key.py              [新增]
├─ test_llm_key_status.py        [新增]
└─ routes/
   ├─ conftest.py                [新增:client(方案 B)+ mock_agent + llm_key_status fixtures]
   ├─ test_chat_route.py         [新增:M4 改名,避免与历史 tests/api/test_chat.py 重名]
   └─ test_llm_status.py         [新增]

tests/integration/
├─ conftest.py                   [新增:client + app_state fixtures,启动 lifespan]
└─ test_chat_real_llm.py         [新增:合并原 tests/api/test_chat.py 的端到端断言]

pyproject.toml                   [新增:pytest 配置载体,§六.7]
```

**删除文件**:`tests/api/test_chat.py`(M1 方案 W,合并到 `tests/integration/test_chat_real_llm.py`)。
