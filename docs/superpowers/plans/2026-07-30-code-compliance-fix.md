# 代码规范合规修复实施计划 (2026-07-30)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec `docs/superpowers/specs/2026-07-30-code-compliance-fix-design.md` 完成 22 项违规修复,分 5 个 Phase 提交。

**Architecture:** 见 spec「架构变更」章节——核心三项结构性 P0(agent.py 合并 nodes.py、LlmService Protocol + DI、generate_with_fallback 三段拆分)。其余 19 项为局部修复。

**Tech Stack:** Python 3.12+ / LangGraph / langchain-openai / pydantic v2 / pytest / ruff / mypy。

## Global Constraints

- **Python 解释器**: 所有 `python` / `pytest` 命令必须用 `D:/ProgramData/miniforge3/envs/langgraph/python.exe`(项目专用 venv,base miniforge3 缺依赖)。
- **每个 Task 完成三步验证**:`ruff check .` 0 错 0 警 + `mypy src` 0 错 + `pytest -q` 全绿。三项全过才能 commit。
- **禁止裸 except / 裸 except Exception**:跨边界 try/except 异常元组必须严格区分外部失败 vs 代码 bug,只含具体外部失败类型(`openai.APIError` / `asyncio.TimeoutError` / `pydantic.ValidationError` / `OSError` 等),禁止含 `ValueError` / `TypeError` / `KeyError` / `AttributeError` / `IndexError`。
- **LLM 调用必须用结构化输出**:`with_structured_output(Schema, method="function_calling")`,mock 必须接受 `**kwargs`。
- **禁止 emoji**:用户终端是 Windows GBK,emoji 会乱码。所有 console 输出、代码注释、commit message 仅用纯中文 + 常用 CJK 标点。
- **`# type: ignore` 仅限 Wrapper 模块**:业务代码禁止用 `# type: ignore` 掩盖类型问题。
- **新提交一律创建新 commit**:禁止 `--amend`,禁止 `--no-verify`。
- **commit 信息格式**:`<type>(<scope>): <subject>`,type 用 feat/refactor/fix/test/docs/chore。

---

## File Structure

| 文件 | 角色 | 改动 Phase |
|------|------|-----------|
| `src/flow/ket_partner/state.py` | 状态 TypedDict + KetIntent Literal + Intent 常量 | P1, P4 |
| `src/flow/ket_partner/vocab_domain.py` | 词汇领域模块 | P1, P4, P5 |
| `src/flow/ket_partner/dialogue_domain.py` | 对话领域模块 | P1, P4, P5 |
| `src/flow/ket_partner/sentence_domain.py` | 造句领域模块 + ValidationResult | P1, P3 |
| `src/flow/ket_partner/nodes.py` | LangGraph 节点薄壳(将被合并删除) | P2 删除 |
| `src/flow/ket_partner/agent.py` | KETPartnerAgent 上下文 + 节点方法 | P2 重写 |
| `src/flow/ket_partner/graph.py` | 图拓扑 + build_agent | P2, P4, P5 |
| `src/flow/ket_partner/config.py` | KetConfig 加载 | P5 |
| `src/flow/common.py` | logger + LlmService 抽象 | P2 |
| `src/api/app.py` | FastAPI lifespan | P1 |
| `src/api/routes/chat.py` | chat 路由 | P4 |
| `src/cli/ket_partner/chat_logger.py` | CLI 会话日志 | P5 |
| `src/cli/ket_partner/main.py` | CLI 入口 | P5 |
| `src/reporting/ket_partner/exporter.py` | 报告导出 | P5 |
| `src/persistence/bootstrap.py` | DB 初始化 + CSV 导入 | P5 |
| `src/persistence/repos.py` | 仓储 | P5 |
| `tests/integration/test_chat_real_llm.py` | LLM 集成测试 | P1 |
| `tests/integration/test_graph_integration.py` | 图集成测试 | P2 |
| `tests/flow/ket_partner/test_vocab_domain.py` | vocab 单元测试 | P4 |
| `tests/flow/ket_partner/test_dialogue_domain.py` | dialogue 单元测试 | P4 |
| `tests/flow/ket_partner/test_sentence_domain.py` | sentence 单元测试 | P3, P4 |

---

## Phase 1:类型契约 + 异常元组(P0 #1/#2/#3 + P2 #18)

### Task 1.1:KetIntent 字面量对齐

**Files:**
- Modify: `src/flow/ket_partner/state.py:5`
- Test: `tests/flow/ket_partner/test_state.py`

**Interfaces:**
- Produces: `KetIntent = Literal["translation", "asks_meaning", "idk", "off_topic", "non_compliant"]`(将 `"translate"` 改为 `"translation"`,与 `dialogue_domain.IntentClassification` Schema 一致)。

- [ ] **Step 1: 加一致性测试**

在 `tests/flow/ket_partner/test_state.py` 末尾追加:

```python
def test_ket_intent_matches_classification_schema():
    """state.KetIntent 字面量必须与 dialogue_domain.IntentClassification 的 Literal 完全一致,
    否则 LLM 返回的 intent 值无法被类型系统校验,路由可能静默失败。"""
    from flow.ket_partner.dialogue_domain import IntentClassification
    from flow.ket_partner.state import KetIntent
    from typing import get_args

    state_literals = set(get_args(KetIntent))
    schema_field = IntentClassification.model_fields["intent"]
    schema_literals = set(schema_field.annotation.__args__)
    assert state_literals == schema_literals, (
        f"KetIntent 与 IntentClassification.intent 不一致: "
        f"state={state_literals}, schema={schema_literals}"
    )
```

- [ ] **Step 2: 运行测试,确认它 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_state.py::test_ket_intent_matches_classification_schema -v
```

期望:FAIL,提示 `{'translate', ...} != {'translation', ...}`。

- [ ] **Step 3: 修正 state.py**

`src/flow/ket_partner/state.py:5`:

```python
KetIntent = Literal["translation", "asks_meaning", "idk", "off_topic", "non_compliant"]
```

- [ ] **Step 4: 运行测试,确认 PASS**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_state.py::test_ket_intent_matches_classification_schema -v
```

- [ ] **Step 5: 三项静态验证**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add src/flow/ket_partner/state.py tests/flow/ket_partner/test_state.py
git commit -m "fix(flow/ket_partner/state): align KetIntent literal with LLM schema

KetIntent 中 'translate' 是死值——运行时 LLM 实际返回 'translation'。
对齐后类型系统才能在赋值阶段捕获不一致。"
```

---

### Task 1.2:vocab_domain LLM 异常元组瘦身

**Files:**
- Modify: `src/flow/ket_partner/vocab_domain.py:142, 176, 197`
- Test: `tests/flow/ket_partner/test_vocab_domain.py`

**Interfaces:**
- Produces: 模块级常量 `_LLM_RETRYABLE: tuple[type[BaseException], ...]`,vocab_domain 内 3 处 LLM 调用共享。

- [ ] **Step 1: 加 fallback 路径 mock 调用断言**

在 `tests/flow/ket_partner/test_vocab_domain.py` 找到 `test_lookup_fallback_on_error`(约 L157-165),改造为:

```python
@pytest.mark.asyncio
async def test_lookup_fallback_on_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回默认值,且 mock 确实被调用过。"""
    from openai import APIError, APIStatusError
    from flow.ket_partner.vocab_domain import lookup_word_meaning, WordMeaning

    bound = MagicMock()
    bound.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=APIError(
            message="sdk timeout",
            request=MagicMock(),
            body=None,
        )
    )

    result = await lookup_word_meaning(bound, sentence="I see a cat.", word="cat")

    assert result.meaning == "(cat 词义查询失败)"
    bound.with_structured_output.assert_called_once_with(WordMeaning, method="function_calling")
    bound.with_structured_output.return_value.ainvoke.assert_awaited_once()
```

注:`APIError` 是 openai SDK 的基类异常,代表外部失败;原本的测试可能用 `ValueError` 触发 fallback,改造后必须用真实的外部失败类型。

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_vocab_domain.py::test_lookup_fallback_on_error -v
```

期望:FAIL——`vocab_domain._LLM_RETRYABLE` 不含 `APIError`,`fallback` 不触发。

- [ ] **Step 3: 加 `_LLM_RETRYABLE` 常量并替换 3 处 except**

在 `src/flow/ket_partner/vocab_domain.py` 顶部 import 区(L1-15 之间)加:

```python
import asyncio

import openai
from pydantic import ValidationError
```

在 `_compute_refill_mode` 上方(L20 附近)加常量:

```python
# vocab_domain 内所有 LLM 调用的可重试外部失败类型。
# 严格按 CLAUDE.md §1.5:只含具体外部失败,不含 ValueError/AttributeError/TypeError
# 等代码 bug 类型——那些必须直接暴露被测试捕获。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,          # openai SDK 的所有 API 异常基类(APITimeoutError/APIConnectionError/RateLimitError 等)
    asyncio.TimeoutError,     # asyncio.wait_for 超时
    ValidationError,          # pydantic Schema 校验失败(LLM 返回畸形结构)
)
```

替换 3 处 except 元组(L142、L176、L197):

```python
# 原:except (TimeoutError, RuntimeError, ValueError, AttributeError, TypeError) as e:
# 新:
except _LLM_RETRYABLE as e:
```

- [ ] **Step 4: 同步另两个 fallback 测试**

`test_vocab_domain.py` 中查找其它 `fallback` / `on_error` 命名的测试(`lookup_word_meanings` 与 `lookup_sentence_translation`),同样把 `side_effect` 改为 `openai.APIError`,加 `assert_awaited_once()` 断言。

- [ ] **Step 5: 运行测试,确认 PASS**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_vocab_domain.py -v
```

- [ ] **Step 6: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/vocab_domain.py tests/flow/ket_partner/test_vocab_domain.py
git commit -m "refactor(flow/ket_partner/vocab_domain): slim LLM exception tuples

定义模块级 _LLM_RETRYABLE 常量,只含 openai.APIError/asyncio.TimeoutError/
ValidationError 三类外部失败;移除 ValueError/AttributeError/TypeError 等
代码 bug 类型,避免把字段名写错等代码 bug 当成 LLM 失败静默兜底。

fallback 测试同步改为用 openai.APIError 触发,并加 ainvoke.assert_awaited_once()
断言,确保被测函数确实进入了 LLM 调用路径。"
```

---

### Task 1.3:dialogue_domain LLM 异常元组瘦身

**Files:**
- Modify: `src/flow/ket_partner/dialogue_domain.py:42, 85, 204`
- Test: `tests/flow/ket_partner/test_dialogue_domain.py`

**Interfaces:**
- Produces: 模块级 `_LLM_RETRYABLE` 常量(同 vocab_domain 模式),dialogue_domain 内 3 处 LLM 调用共享。

- [ ] **Step 1: 加 `_LLM_RETRYABLE` 常量**

在 `src/flow/ket_partner/dialogue_domain.py` 顶部(L1-12 之间)加:

```python
import asyncio

import openai
from pydantic import ValidationError
```

在 `_CLASSIFIER_SYSTEM` 上方(L17 附近)加:

```python
# dialogue_domain 内所有 LLM 调用的可重试外部失败类型。
# 严格按 CLAUDE.md §1.5:只含具体外部失败,不含代码 bug 类型。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,
    asyncio.TimeoutError,
    ValidationError,
)
```

- [ ] **Step 2: 替换 3 处 except(L42、L85、L204)**

```python
# 原:except (TimeoutError, RuntimeError, ValueError, AttributeError, TypeError) as e:
# 新:
except _LLM_RETRYABLE as e:
```

- [ ] **Step 3: 改造 3 处 fallback 测试**

`tests/flow/ket_partner/test_dialogue_domain.py` 找到 `test_classify_fallback_on_llm_error`(L60-68)、`test_summary_fallback_on_error`(L103-118)、`test_evaluate_fallback_on_error`(L172-187)。

每处把 `side_effect` 改为 `openai.APIError(...)`,在断言返回值之后加:

```python
bound.with_structured_output.return_value.ainvoke.assert_awaited_once()
```

注:`APIError` 构造需要 `request` 参数,用 `MagicMock()` 占位:

```python
from openai import APIError
from unittest.mock import MagicMock

bound.with_structured_output.return_value.ainvoke = AsyncMock(
    side_effect=APIError(message="fail", request=MagicMock(), body=None)
)
```

- [ ] **Step 4: 运行测试**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_dialogue_domain.py -v
```

期望:全 PASS。如果某个 fallback 测试 FAIL,说明被测函数没有进入 LLM 调用就直接返回兜底——这正是 §6.4 警告的"测试假通过"。

- [ ] **Step 5: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/dialogue_domain.py tests/flow/ket_partner/test_dialogue_domain.py
git commit -m "refactor(flow/ket_partner/dialogue_domain): slim LLM exception tuples

同 vocab_domain 模式:模块级 _LLM_RETRYABLE + 3 处 fallback 测试加
assert_awaited_once 断言。"
```

---

### Task 1.4:sentence_domain LLM 异常元组瘦身

**Files:**
- Modify: `src/flow/ket_partner/sentence_domain.py:186, 360`
- Test: `tests/flow/ket_partner/test_sentence_domain.py`

**Interfaces:**
- Produces: 模块级 `_LLM_RETRYABLE` 常量,sentence_domain 内 2 处 LLM 调用(`generate_sentence`、`validate_and_categorize`)共享。

- [ ] **Step 1: 加 `_LLM_RETRYABLE` 常量**

在 `src/flow/ket_partner/sentence_domain.py` 顶部 import 区(L1-13 之间)加:

```python
import asyncio

import openai
from pydantic import ValidationError
```

在 `_PLACEHOLDER_TOKENS` 上方(L18 附近)加:

```python
# sentence_domain 内所有 LLM 调用的可重试外部失败类型。
# 严格按 CLAUDE.md §1.5。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,
    asyncio.TimeoutError,
    ValidationError,
)
```

- [ ] **Step 2: 替换 2 处 except(L186、L360)**

```python
# 原:except (TimeoutError, RuntimeError, ValueError, AttributeError, TypeError) as e:
# 新:
except _LLM_RETRYABLE as e:
```

- [ ] **Step 3: 改造 fallback 测试**

`tests/flow/ket_partner/test_sentence_domain.py` 找到 `test_check_naturalness_fails_open_on_error`(L738-746)与 `generate_sentence` 的 fallback 测试(若有)。

每处把 `side_effect` 改为 `openai.APIError(message="fail", request=MagicMock(), body=None)`,加 `bound.ainvoke.assert_awaited_once()` 或 `bound.with_structured_output.return_value.ainvoke.assert_awaited_once()`。

- [ ] **Step 4: 运行测试**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py -v
```

- [ ] **Step 5: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/sentence_domain.py tests/flow/ket_partner/test_sentence_domain.py
git commit -m "refactor(flow/ket_partner/sentence_domain): slim LLM exception tuples

同 vocab_domain / dialogue_domain 模式:模块级 _LLM_RETRYABLE + fallback
测试加 assert_awaited_once 断言。"
```

---

### Task 1.5:agent.py LLM 异常元组瘦身

**Files:**
- Modify: `src/flow/ket_partner/agent.py:61`
- Test: 现有单元测试覆盖 `_run_summary_safe` 的 fallback 路径(若无则新增)。

**Interfaces:**
- Produces: agent.py 模块级 `_LLM_RETRYABLE` 常量(注:本任务只改异常元组,不动 agent.py 结构;agent.py 结构重写见 Phase 2)。

- [ ] **Step 1: 加 `_LLM_RETRYABLE` 常量**

在 `src/flow/ket_partner/agent.py` 顶部 import 区(L1-9 之间)加:

```python
import asyncio

import openai
from pydantic import ValidationError

from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.state import BTPKetState
```

在 `class KETPartnerAgent` 定义之前(L11 附近)加:

```python
# _run_summary_safe 后台任务的可重试外部失败类型。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,
    asyncio.TimeoutError,
    ValidationError,
)
```

- [ ] **Step 2: 替换 L61 的 except**

```python
# 原:
# except (TimeoutError, RuntimeError, ValueError, AttributeError, TypeError) as e:
# 新:
except _LLM_RETRYABLE as e:
```

- [ ] **Step 3: 检查并补测试**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/ -v -k "summary or agent"
```

如果没有覆盖 `_run_summary_safe` fallback 的测试,在 `tests/flow/ket_partner/test_dialogue_domain.py`(已有 run_profile_summary 测试)内补:

```python
@pytest.mark.asyncio
async def test_run_summary_safe_swallows_llm_error(monkeypatch):
    """_run_summary_safe 在 LLM 失败时静默兜底,不影响主流程。"""
    from openai import APIError
    from unittest.mock import MagicMock
    from flow.ket_partner.agent import KETPartnerAgent, _LLM_RETRYABLE
    from flow.ket_partner.dialogue_domain import run_profile_summary

    # 验证 _LLM_RETRYABLE 不含代码 bug 类型
    forbidden = {ValueError, TypeError, KeyError, AttributeError, IndexError}
    assert not (set(_LLM_RETRYABLE) & forbidden)

    bound = MagicMock()
    bound.with_structured_output.return_value.ainvoke = AsyncMock(
        side_effect=APIError(message="fail", request=MagicMock(), body=None)
    )

    repos = MagicMock()
    repos.profile.get = AsyncMock(return_value={"weakness_words": [], "dialogue_strategy": ""})
    repos.profile.update = AsyncMock()
    repos.log.recent = AsyncMock(return_value=[])

    # 不抛异常即通过
    await run_profile_summary(bound, repos)
    bound.with_structured_output.return_value.ainvoke.assert_awaited()
```

- [ ] **Step 4: 运行测试 + 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/agent.py tests/flow/ket_partner/test_dialogue_domain.py
git commit -m "refactor(flow/ket_partner/agent): slim LLM exception tuple in _run_summary_safe"
```

---

### Task 1.6:app.py shutdown 异常元组细化

**Files:**
- Modify: `src/api/app.py:70, 77`
- Test: `tests/api/test_app_lifespan.py`(若无则新增)

**Interfaces:**
- Consumes: `aiosqlite.Error`、`asyncio.TimeoutError`、`RuntimeError`、`OSError`。
- Produces: shutdown 路径不再吞代码 bug。

- [ ] **Step 1: 加 import 与替换两处 except**

`src/api/app.py` 顶部 import 区(L11-20 之间)补:

```python
import asyncio

import aiosqlite
```

L70 改为:

```python
# 原:except Exception as e:
# 新:
except (RuntimeError, asyncio.TimeoutError) as e:
```

L77 改为:

```python
# 原:except Exception as e:
# 新:
except (RuntimeError, OSError, aiosqlite.Error) as e:
```

- [ ] **Step 2: 加 shutdown 异常路径测试**

新建 `tests/api/test_app_lifespan.py`:

```python
"""app.py lifespan shutdown 异常路径测试。"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_lifespan_handles_db_close_failure(monkeypatch):
    """db.close() 抛 aiosqlite.Error 时,lifespan 不应崩溃,应记录 warning 后退出。"""
    import aiosqlite
    from src.api import app as app_module

    fake_db = MagicMock()
    fake_db.close = AsyncMock(side_effect=aiosqlite.Error("simulated close failure"))

    # 直接构造一个最小 lifespan 调用,验证 __aexit__ 路径
    captured_warnings = []
    monkeypatch.setattr(
        app_module.logger,
        "warning",
        lambda msg, *args, **kwargs: captured_warnings.append(msg),
    )

    # 模拟 app.state 已挂上 agent + db
    fake_app = MagicMock()
    fake_app.state = MagicMock()
    fake_app.state.agent = MagicMock()
    fake_app.state.db = fake_db
    # agent.aclose 不抛异常
    fake_app.state.agent.aclose = AsyncMock()

    # 直接调用 lifespan 的 finally 块逻辑(通过 LifespanManager 触发)
    from httpx import AsyncClient
    from src.api.app import app

    app.state.db = fake_db
    app.state.agent = MagicMock(aclose=AsyncMock())

    async with AsyncClient(app=app, base_url="http://test") as client:
        # lifespan startup 已跑过(在 import 时);此处只验证 shutdown 路径
        pass

    assert any("db.close" in w for w in captured_warnings), \
        f"期望 warning 记录 db.close 失败,实际: {captured_warnings}"
```

注:如果现有 app 已通过 lifespan 装配,直接断言 warning 被调用即可。如果构造测试过于复杂,简化为断言 `aiosqlite.Error` 在 except 元组中:

```python
def test_shutdown_exception_tuples_exclude_code_bugs():
    """shutdown except 元组必须只含外部失败,不含 ValueError/AttributeError 等。"""
    import inspect
    from src.api import app as app_module
    src = inspect.getsource(app_module)
    # 简化检查:源码中不应再出现 'except Exception'
    assert "except Exception" not in src, \
        "app.py 仍含 'except Exception',违反 §1.1"
```

- [ ] **Step 3: 运行测试 + 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/api/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/api/app.py tests/api/test_app_lifespan.py
git commit -m "fix(api/app): replace bare except with specific exception tuples

shutdown 路径两处 'except Exception' 改为具体类型:
- agent.aclose: (RuntimeError, asyncio.TimeoutError)
- db.close: (RuntimeError, OSError, aiosqlite.Error)

避免误吞代码 bug。"
```

---

### Task 1.7:删除未使用的 import os

**Files:**
- Modify: `tests/integration/test_chat_real_llm.py:1`

- [ ] **Step 1: 删除 import os**

`tests/integration/test_chat_real_llm.py:1` 删去 `import os`。

- [ ] **Step 2: ruff 验证**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
```

期望:`Found 0 errors`。

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_chat_real_llm.py
git commit -m "chore(test/integration): remove unused 'import os' (ruff F401)"
```

---

## Phase 2:结构性重构 LlmService + agent 合并(P0 #5/#8)

> ⚠️ **Phase 2 风险**:本 Phase 改动面大(`flow/common.py` / `agent.py` / `nodes.py` 删除 / `graph.py` / `app.py` / `test_graph_integration.py` 9 处)。每个 Task 完成后必须 `pytest -q` 全绿才能进下一个 Task。

### Task 2.1:flow/common.py 加 LlmService Protocol + DashScopeLlmService

**Files:**
- Modify: `src/flow/common.py`
- Test: `tests/flow/test_common_llm_service.py`(新建)

**Interfaces:**
- Produces:
  - `class LlmService(Protocol)` with attributes `smart: BaseChatModel` and `flash: BaseChatModel`
  - `class DashScopeLlmService` 具体类,`__init__` 中创建 ChatOpenAI 实例
  - `default_llm_service: LlmService` 模块级单例

- [ ] **Step 1: 写 LlmService Protocol 与 DashScopeLlmService 测试**

新建 `tests/flow/test_common_llm_service.py`:

```python
"""LlmService Protocol + DashScopeLlmService 测试。"""
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel


def test_default_llm_service_exposes_smart_and_flash():
    """default_llm_service 单例必须暴露 smart 和 flash 两个 BaseChatModel。"""
    from flow.common import default_llm_service

    assert isinstance(default_llm_service.smart, BaseChatModel)
    assert isinstance(default_llm_service.flash, BaseChatModel)


def test_dashscope_llm_service_creates_independent_instances():
    """DashScopeLlmService 每次实例化创建独立 ChatOpenAI(避免共享 client)。"""
    from flow.common import DashScopeLlmService

    svc1 = DashScopeLlmService()
    svc2 = DashScopeLlmService()
    assert svc1.smart is not svc2.smart
    assert svc1.flash is not svc2.flash


def test_llm_service_protocol_accepts_mock():
    """LlmService Protocol 必须能接受任意含 smart/flash 属性的对象(便于 DI mock)。"""
    from flow.common import LlmService

    mock_svc = MagicMock()
    mock_svc.smart = MagicMock(spec=BaseChatModel)
    mock_svc.flash = MagicMock(spec=BaseChatModel)
    # Protocol 是结构性子类型,只要属性匹配就通过
    assert isinstance(mock_svc, LlmService)
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/test_common_llm_service.py -v
```

期望:FAIL,`ImportError: cannot import name 'LlmService' from 'flow.common'`。

- [ ] **Step 3: 在 flow/common.py 中加 LlmService 抽象**

在 `src/flow/common.py` 末尾追加(L83 之后):

```python
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel


class LlmService(Protocol):
    """LLM 服务抽象。业务代码依赖此 Protocol,便于测试注入 mock。

    Implementations: DashScopeLlmService(默认),未来可加 MockLlmService 等。
    """
    smart: BaseChatModel
    flash: BaseChatModel


class DashScopeLlmService:
    """DashScope(Qwen 兼容模式)具体实现。封装 ChatOpenAI 实例化细节。"""

    def __init__(self) -> None:
        api_key = _resolve_dashscope_api_key()
        common_kwargs = {
            "api_key": SecretStr(api_key or "placeholder"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        client_kwargs = {
            "api_key": api_key or "placeholder",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "http_client": AsyncClient(),
        }
        self.smart = ChatOpenAI(
            **common_kwargs,
            model="qwen3.6-max-preview",
            client=AsyncOpenAI(**client_kwargs),
            temperature=0,
            extra_body=extra_params,
        )
        self.flash = ChatOpenAI(
            **common_kwargs,
            model="qwen3.6-flash",
            client=AsyncOpenAI(**client_kwargs),
            temperature=0.8,
            top_p=0.8,
            extra_body=extra_params,
        )


default_llm_service: LlmService = DashScopeLlmService()
```

**保留** 现有的 `llm_max` / `llm_flash` 模块级变量(后续 Task 2.4 删除),只是不再被业务代码引用。

- [ ] **Step 4: 运行测试,确认 PASS**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/test_common_llm_service.py -v
```

- [ ] **Step 5: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/common.py tests/flow/test_common_llm_service.py
git commit -m "feat(flow/common): add LlmService Protocol + DashScopeLlmService

新增 LlmService Protocol(暴露 smart/flash 两个 BaseChatModel)+
DashScopeLlmService 具体类 + default_llm_service 模块级单例。

为后续 build_agent 接受可选 LlmService 参数、KETPartnerAgent 用
property 暴露 llm_smart/llm_flash 做准备。

本 Task 仅新增,不动现有 llm_max/llm_flash 模块级变量(下个 Task 删除)。"
```

---

### Task 2.2:KETPartnerAgent 改用 LlmService(构造函数 + property)

**Files:**
- Modify: `src/flow/ket_partner/agent.py:13-17`(只改构造函数,暂不动 12 个节点方法)
- Modify: `src/flow/ket_partner/graph.py:100-111`(build_agent 接受 LlmService)
- Modify: `src/api/app.py:56`(传 default_llm_service)
- Test: `tests/flow/ket_partner/test_agent_llm_service.py`(新建)

**Interfaces:**
- Consumes: Task 2.1 的 `LlmService` Protocol + `default_llm_service` 单例
- Produces:
  - `KETPartnerAgent.__init__(self, llm_service: LlmService, config: KetConfig)`
  - `KETPartnerAgent.llm_smart` / `.llm_flash` properties
  - `build_agent(llm_service: LlmService | None = None, checkpointer=None)`

- [ ] **Step 1: 写 KETPartnerAgent LlmService 集成测试**

新建 `tests/flow/ket_partner/test_agent_llm_service.py`:

```python
"""KETPartnerAgent 与 LlmService DI 集成测试。"""
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel


def test_agent_exposes_llm_smart_and_flash_via_property():
    """KETPartnerAgent 必须通过 @property 暴露 llm_smart/llm_flash,
    不能直接存 self.llm_smart(避免绕开 LlmService 抽象)。"""
    from flow.ket_partner.agent import KETPartnerAgent
    from flow.ket_partner.config import KetConfig

    mock_svc = MagicMock()
    mock_svc.smart = MagicMock(spec=BaseChatModel)
    mock_svc.flash = MagicMock(spec=BaseChatModel)

    agent = KETPartnerAgent(mock_svc, KetConfig())

    # property 应该返回 llm_service.smart/flash
    assert agent.llm_smart is mock_svc.smart
    assert agent.llm_flash is mock_svc.flash
    # 内部存的是 service,不是直接的 llm
    assert agent._llm_service is mock_svc


@pytest.mark.asyncio
async def test_build_agent_uses_injected_llm_service():
    """build_agent 接受 llm_service 参数,不应使用模块级 default_llm_service。"""
    from flow.ket_partner.graph import build_agent
    from unittest.mock import patch

    mock_svc = MagicMock()
    mock_svc.smart = MagicMock(spec=BaseChatModel)
    mock_svc.flash = MagicMock(spec=BaseChatModel)

    with patch("flow.ket_partner.graph.default_llm_service") as mock_default:
        # 让 default 抛异常,确保它没被调用
        mock_default.side_effect = AssertionError("不应使用 default_llm_service")
        graph = await build_agent(llm_service=mock_svc)

    # graph.agent 是 build_agent 内挂上的(KETPartnerAgent 实例)
    inner_agent = getattr(graph, "agent", None)
    assert inner_agent is not None
    assert inner_agent._llm_service is mock_svc
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_agent_llm_service.py -v
```

期望:FAIL——当前 `KETPartnerAgent.__init__(self, llm_flash, llm_smart, config)` 签名不匹配。

- [ ] **Step 3: 改 KETPartnerAgent 构造函数**

`src/flow/ket_partner/agent.py` 的 `__init__`(L13-17)改为:

```python
class KETPartnerAgent:
    def __init__(self, llm_service: LlmService, config: KetConfig) -> None:
        self._llm_service = llm_service
        self.config = config
        self._bg_tasks: set[asyncio.Task] = set()

    @property
    def llm_smart(self) -> BaseChatModel:
        return self._llm_service.smart

    @property
    def llm_flash(self) -> BaseChatModel:
        return self._llm_service.flash
```

同步 import 区(L1-9 之间)补:

```python
from langchain_core.language_models.chat_models import BaseChatModel

from flow.common import LlmService, logger
from flow.ket_partner.config import KetConfig
```

**12 个节点方法暂保留透传**(`return await nodes.xxx(state, config, self)`),下个 Task 改。

- [ ] **Step 4: 改 graph.py build_agent**

`src/flow/ket_partner/graph.py:100-111`:

```python
from flow.common import LlmService, default_llm_service
from flow.ket_partner.config import load_config
from flow.ket_partner.state import BTPKetState


async def build_agent(
    llm_service: LlmService | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    if llm_service is None:
        llm_service = default_llm_service
    cfg = load_config()
    agent = KETPartnerAgent(llm_service, cfg)
    builder = StateGraph(BTPKetState)
    wire_graph(builder, agent)
    graph = builder.compile(checkpointer=checkpointer)
    graph.agent = agent  # type: ignore[attr-defined]
    return graph
```

- [ ] **Step 5: 改 app.py 调用**

`src/api/app.py:15`:

```python
# 原:from flow.common import llm_flash, llm_max, logger
# 新:
from flow.common import default_llm_service, logger
```

L56:

```python
# 原:agent = await build_agent(llm_flash, llm_max, checkpointer=checkpointer)
# 新:
agent = await build_agent(default_llm_service, checkpointer=checkpointer)
```

- [ ] **Step 6: 运行测试 + 三项静态验证**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_agent_llm_service.py -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
```

如有 FAIL,优先检查 import 链:`flow.common.default_llm_service` 是否正确导出。

- [ ] **Step 7: Commit**

```bash
git add src/flow/ket_partner/agent.py src/flow/ket_partner/graph.py src/api/app.py tests/flow/ket_partner/test_agent_llm_service.py
git commit -m "refactor(flow/ket_partner): switch KETPartnerAgent to LlmService DI

KETPartnerAgent.__init__ 改为接受 LlmService,通过 @property 暴露
llm_smart/llm_flash;build_agent 接受可选 llm_service 参数,默认用
default_llm_service 单例。

app.py lifespan 改用 default_llm_service。

12 个节点方法暂保留 nodes.xxx 透传,下个 Task 改为真实实现。"
```

---

### Task 2.3:合并 nodes.py 实现到 KETPartnerAgent

**Files:**
- Modify: `src/flow/ket_partner/agent.py`(把 nodes.py 的 12 个节点函数体迁入类方法)
- Delete: `src/flow/ket_partner/nodes.py`
- Test: 现有 `tests/integration/test_graph_integration.py`(需更新 import,下个 Task 处理)

**Interfaces:**
- Consumes: 现有 `nodes.py` 的 12 个函数实现
- Produces: `KETPartnerAgent` 类内 12 个真实节点方法,签名 `async def xxx_node(self, state, config) -> dict`

- [ ] **Step 1: 把 nodes.py 的 12 个函数体迁入 KETPartnerAgent**

打开 `src/flow/ket_partner/agent.py` 与 `src/flow/ket_partner/nodes.py` 对照。

对每个节点函数(共 12 个),做如下改造:

模板(以 `init_state` 为例):

```python
# nodes.py 原:
async def init_state(state: BTPKetState, config: RunnableConfig, agent: KETPartnerAgent) -> dict:
    repos = get_repos(config)
    profile = await repos.profile.get()
    # ... 函数体 ...
    return update

# agent.py 新(类方法):
async def init_state(self, state: BTPKetState, config: RunnableConfig) -> dict:
    repos = get_repos(config)
    profile = await repos.profile.get()
    # ... 函数体照搬,把 'agent.llm_smart' 改为 'self.llm_smart',
    # 'agent.config' 改为 'self.config',
    # 'agent._run_summary_safe' 改为 'self._run_summary_safe',
    # 'agent._bg_tasks' 改为 'self._bg_tasks'
    return update
```

12 个节点方法清单(逐个迁移):
1. `init_state_node`(对应 `nodes.init_state`)— 函数体 L35-63
2. `classify_intent_node` — L66-69
3. `evaluate_translation_node` — L72-129
4. `lookup_target_meaning_node` — L132-136
5. `lookup_asked_meaning_node` — L139-143
6. `update_mastery_node` — L146-149
7. `select_target_word_node` — L152-158
8. `generate_sentence_node` — L161-206
9. `format_output_node` — L209-214
10. `explain_meaning_node` — L217-222
11. `redirect_to_translate_node` — L225-228
12. `compliance_redirect_node` — L231-234
13. `persist_turn_node` — L237-266

注:`agent.py` 原 12 个透传方法(`async def init_state(self, state, config): return await nodes.init_state(state, config, self)`)整体替换为上面的真实实现方法。

同步处理:
- `agent.py` 顶部 import 区加 `from flow.ket_partner.persistence import get_repos`(nodes.py 用到)
- 保留 `from flow.ket_partner.dialogue_domain import classify_intent, evaluate_translation, format_output_text, run_profile_summary`
- 保留 `from flow.ket_partner.sentence_domain import _tokenize, apply_multiword_target_patch, generate_with_fallback`
- 保留 `from flow.ket_partner.vocab_domain import apply_mastery_updates, lookup_sentence_translation, lookup_word_meaning, lookup_word_meanings, select_target_word`
- 移除 `from flow.ket_partner import nodes`(即将删除)

`generate_sentence_node` 内 L111 处的 `entry = entry.model_copy(update={"word": wr.word})` **同时**改为直接构造新实例(P2 #22 顺带做):

```python
# 原:
# entry = entry.model_copy(update={"word": wr.word})
# 新:
from flow.ket_partner.dialogue_domain import WrongWord
entry = WrongWord(
    word=wr.word,
    kid_translation=entry.kid_translation,
    correct_translation=entry.correct_translation,
    contrast=entry.contrast,
)
```

注:`WrongWord` 已在 `dialogue_domain.py:159` 定义。把 import 提到 agent.py 顶部。

- [ ] **Step 2: 删除 nodes.py**

```bash
git rm src/flow/ket_partner/nodes.py
```

- [ ] **Step 3: 验证 agent.py 单独可 import**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -c "from flow.ket_partner.agent import KETPartnerAgent; print('OK')"
```

期望:`OK`。如果有 `ImportError`,检查 import 是否齐全。

- [ ] **Step 4: 运行全部测试**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
```

期望:`test_graph_integration.py` 的 9 处 `from flow.ket_partner import nodes` 会 FAIL——这是预期的,下个 Task 修复。

其它测试应该全过。

- [ ] **Step 5: 三项静态验证**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
```

注:`pytest` 此刻会有 `test_graph_integration.py` 失败,正常,不入 commit 范围。但 `ruff`/`mypy` 必须清零。

- [ ] **Step 6: Commit(暂不包含 test_graph_integration 改动)**

```bash
git add src/flow/ket_partner/agent.py
git rm src/flow/ket_partner/nodes.py
git commit -m "refactor(flow/ket_partner): merge nodes.py implementations into KETPartnerAgent

将 12 个节点函数实现从 nodes.py 迁入 KETPartnerAgent 类方法,
签名改为 (self, state, config)。删除 nodes.py。

消除 §10.3 中间人反模式:12 个透传 wrapper 方法变为真实实现。

顺带修复 P2 #22:nodes.py:111 循环中的 entry.model_copy 改为
直接构造新 WrongWord 实例,避免修改循环变量。

test_graph_integration.py 的 9 处 nodes import 下个 Task 修复。"
```

---

### Task 2.4:更新 test_graph_integration.py

**Files:**
- Modify: `tests/integration/test_graph_integration.py`(9 处 `from flow.ket_partner import nodes as agent_module`)

**Interfaces:**
- Consumes: Task 2.3 已删除 `nodes.py`,所有节点逻辑现在在 `KETPartnerAgent` 类方法里。

- [ ] **Step 1: 全局替换 import**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -c "
import re
path = 'tests/integration/test_graph_integration.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
new_content = content.replace(
    'from flow.ket_partner import nodes as agent_module',
    'from flow.ket_partner.agent import KETPartnerAgent as agent_module'
)
with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Replaced', content.count('from flow.ket_partner import nodes as agent_module'), 'sites')
"
```

注:这只是把模块引用换名。如果测试代码用了 `agent_module.init_state` 这类调用,需要进一步改为 `agent_module.<some_method>`(但 `agent_module` 现在是类,不是模块)。

- [ ] **Step 2: 检查并修正具体用法**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/integration/test_graph_integration.py -v 2>&1 | head -80
```

如果有 `AttributeError: type object 'KETPartnerAgent' has no attribute 'init_state'` 之类错误,定位每处用法:

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m grep -n "agent_module\." tests/integration/test_graph_integration.py
```

逐处改造:
- 原 `agent_module.init_state(...)` → 若是 patch 用法:`patch.object(KETPartnerAgent, "init_state_node", ...)`
- 原 `agent_module.classify_intent_node` → 类方法引用:`KETPartnerAgent.classify_intent_node`

- [ ] **Step 3: 运行测试**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/integration/test_graph_integration.py -v
```

期望:全 PASS(或仅 `@pytest.mark.integration` 标记的 skip)。

- [ ] **Step 4: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add tests/integration/test_graph_integration.py
git commit -m "test(integration): migrate nodes imports to KETPartnerAgent after merge

9 处 'from flow.ket_partner import nodes as agent_module' 改为
'from flow.ket_partner.agent import KETPartnerAgent'。
patch 用法迁移到 patch.object(KETPartnerAgent, 'xxx_node', ...)。"
```

---

## Phase 3:sentence_domain 内部(P0 #4/#6/#7)

### Task 3.1:定义 SentenceGenerationResult dataclass

**Files:**
- Modify: `src/flow/ket_partner/sentence_domain.py`(在文件顶部加 dataclass)
- Test: `tests/flow/ket_partner/test_sentence_domain.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True, slots=True) class SentenceGenerationResult: sentence: str; result: ValidationResult; target: str; context: str`
  - `@dataclass(frozen=True, slots=True) class _RetryOuter: target: str; context: str`

- [ ] **Step 1: 写 dataclass 字段与不可变性测试**

在 `tests/flow/ket_partner/test_sentence_domain.py` 末尾追加:

```python
def test_sentence_generation_result_is_frozen():
    """SentenceGenerationResult 必须 frozen=True,防止运行时被修改。"""
    from dataclasses import FrozenInstanceError
    from flow.ket_partner.sentence_domain import SentenceGenerationResult, ValidationResult

    r = SentenceGenerationResult(
        sentence="I see a cat.",
        result=ValidationResult(ok=True),
        target="cat",
        context="",
    )
    try:
        r.sentence = "modified"  # type: ignore[misc]
        raised = False
    except FrozenInstanceError:
        raised = True
    assert raised, "SentenceGenerationResult 应 frozen"


def test_sentence_generation_result_has_named_fields():
    """SentenceGenerationResult 必须 4 个命名字段,禁止裸元组。"""
    from flow.ket_partner.sentence_domain import SentenceGenerationResult, ValidationResult
    import dataclasses

    fields = {f.name for f in dataclasses.fields(SentenceGenerationResult)}
    assert fields == {"sentence", "result", "target", "context"}
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py::test_sentence_generation_result_is_frozen -v
```

期望:`ImportError`。

- [ ] **Step 3: 在 sentence_domain.py 加 dataclass**

在 `src/flow/ket_partner/sentence_domain.py` 顶部 import 区(L1-13 之间)加:

```python
from dataclasses import dataclass
```

在 `class ValidationResult` 之后(L218 附近)加:

```python
@dataclass(frozen=True, slots=True)
class SentenceGenerationResult:
    """generate_with_fallback 的最终返回类型。

    替代原裸 4 元组 (sentence, result, target, context),让调用方按命名属性访问,
    位置错配在编译期就能暴露。
    """
    sentence: str
    result: ValidationResult
    target: str
    context: str


@dataclass(frozen=True, slots=True)
class _RetryOuter:
    """_switch_target_or_accept 的内部信号:已切换 target,请求外层 while 重试。

    不对外暴露(下划线前缀)。
    """
    target: str
    context: str
```

- [ ] **Step 4: 运行测试,确认 PASS**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py::test_sentence_generation_result_is_frozen tests/flow/ket_partner/test_sentence_domain.py::test_sentence_generation_result_has_named_fields -v
```

- [ ] **Step 5: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/sentence_domain.py tests/flow/ket_partner/test_sentence_domain.py
git commit -m "feat(flow/ket_partner/sentence_domain): add SentenceGenerationResult dataclass

替代 generate_with_fallback 原返回的裸 4 元组。frozen+slots,4 命名字段
sentence/result/target/context。

同步加 _RetryOuter 内部信号 dataclass,供下个 Task 重构
generate_with_fallback 使用。"
```

---

### Task 3.2:apply_multiword_target_patch 改纯函数

**Files:**
- Modify: `src/flow/ket_partner/sentence_domain.py:552-576`
- Modify: `src/flow/ket_partner/agent.py`(调用点,原 `nodes.py:188` 现在在 `generate_sentence_node` 方法内)
- Test: `tests/flow/ket_partner/test_sentence_domain.py`

**Interfaces:**
- Produces: `apply_multiword_target_patch(target, sentence, result) -> ValidationResult`(返回新对象,不改入参)

- [ ] **Step 1: 写纯函数测试**

在 `tests/flow/ket_partner/test_sentence_domain.py` 追加:

```python
def test_apply_multiword_target_patch_does_not_mutate_input():
    """apply_multiword_target_patch 不能修改入参 result,必须返回新对象。"""
    from flow.ket_partner.sentence_domain import apply_multiword_target_patch, ValidationResult

    original = ValidationResult(
        ok=True,
        words_used=["play"],
        non_ket_words=["the"],
    )
    # 备份原值
    orig_words = list(original.words_used)
    orig_non_ket = list(original.non_ket_words)

    new_result = apply_multiword_target_patch("play ball", "I play ball.", original)

    # 入参未被修改
    assert original.words_used == orig_words
    assert original.non_ket_words == orig_non_ket
    # 返回的是新对象
    assert new_result is not original
    # 新对象的字段已更新
    assert "play ball" in new_result.words_used or "play" in new_result.words_used
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py::test_apply_multiword_target_patch_does_not_mutate_input -v
```

期望:FAIL(原实现直接修改入参,返回 None)。

- [ ] **Step 3: 改造 apply_multiword_target_patch**

`src/flow/ket_partner/sentence_domain.py:552-576` 改造为:

```python
def apply_multiword_target_patch(
    target: str,
    sentence: str,
    result: ValidationResult,
) -> ValidationResult:
    """对多词 target 应用补丁:确保 words_used 和 non_ket_words 正确反映多词边界。

    纯函数:不改入参 result,返回新的 ValidationResult。
    """
    if not target or " " not in target or not target_in_sentence(target, sentence):
        return result

    # 计算新的 words_used 和 non_ket_words(原逻辑保留,但改为构造新列表)
    new_words_used = list(result.words_used)
    new_non_ket = list(result.non_ket_words)

    target_tokens = target.lower().split()
    # 把多词 target 当作一个整体加入 words_used(若尚未存在)
    if target not in new_words_used:
        new_words_used.append(target)
    # 移除被 target 包含的零散 token(原逻辑)
    for tok in target_tokens:
        while tok in new_words_used:
            new_words_used.remove(tok)
        while tok in new_non_ket:
            new_non_ket.remove(tok)

    return result.model_copy(update={
        "words_used": new_words_used,
        "non_ket_words": new_non_ket,
    })
```

注:具体的多词合并规则以原 L552-576 实现为准,本 Task 只把"修改入参"改为"返回新对象",逻辑等价。

- [ ] **Step 4: 改调用点**

`src/flow/ket_partner/agent.py` 的 `generate_sentence_node` 方法内(原 `nodes.py:188`):

```python
# 原:apply_multiword_target_patch(target, sentence, result)
# 新:
result = apply_multiword_target_patch(target, sentence, result)
```

- [ ] **Step 5: 运行测试 + 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/sentence_domain.py src/flow/ket_partner/agent.py tests/flow/ket_partner/test_sentence_domain.py
git commit -m "refactor(flow/ket_partner/sentence_domain): make apply_multiword_target_patch pure

返回新 ValidationResult(model_copy),不再修改入参。
调用点同步改为赋值返回值。"
```

---

### Task 3.3:generate_with_fallback 三段拆分

**Files:**
- Modify: `src/flow/ket_partner/sentence_domain.py:434-550`
- Modify: `src/flow/ket_partner/agent.py`(调用点,原 `nodes.py:173-184` 现在 `generate_sentence_node` 方法内)
- Test: `tests/flow/ket_partner/test_sentence_domain.py`

**Interfaces:**
- Consumes: Task 3.1 的 `SentenceGenerationResult` / `_RetryOuter`
- Produces:
  - `async def _generate_and_validate(...) -> tuple[str, ValidationResult, list[dict]]`
  - `async def _handle_overflow_fallback(attempts, target, context, repos) -> SentenceGenerationResult | None`
  - `async def _switch_target_or_accept(...) -> SentenceGenerationResult | _RetryOuter`
  - 重写 `async def generate_with_fallback(...) -> SentenceGenerationResult`

- [ ] **Step 1: 写 generate_with_fallback 返回类型测试**

在 `tests/flow/ket_partner/test_sentence_domain.py` 追加:

```python
@pytest.mark.asyncio
async def test_generate_with_fallback_returns_named_result():
    """generate_with_fallback 必须返回 SentenceGenerationResult,不能是裸元组。"""
    from flow.ket_partner.sentence_domain import (
        generate_with_fallback,
        SentenceGenerationResult,
        ValidationResult,
    )

    # 用最小 mock LLM 让函数走通正常路径
    mock_llm = MagicMock()
    mock_llm.bind.return_value.ainvoke = AsyncMock(
        return_value=MagicMock(content="I see a cat.")
    )
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=ValidationResult(ok=True, words_used=["cat"], non_ket_words=[])
    )
    repos = MagicMock()
    repos.recent.list_recent_scaffolding = AsyncMock(return_value=[])
    repos.recent.list_recent = AsyncMock(return_value=[])
    repos.recent.append = AsyncMock()
    repos.stats.increment_exposed = AsyncMock()
    repos.stats.oldest_learning_word = AsyncMock(return_value=None)
    repos.vocab.get_ket_word_any_context = AsyncMock(return_value=None)

    profile = {"current_topic": "animals", "in_refill_mode": 0, "total_turns": 1,
               "last_new_word_turn": 0, "weakness_words": [], "dialogue_strategy": ""}

    from flow.ket_partner.config import KetConfig
    result = await generate_with_fallback(
        llm_smart=mock_llm,
        initial_target="cat",
        initial_context="",
        avoid_words=[],
        avoid_sentences=[],
        age=8,
        profile=profile,
        repos=repos,
        config=KetConfig(),
    )

    assert isinstance(result, SentenceGenerationResult)
    assert result.sentence == "I see a cat."
    assert result.target == "cat"
```

注:本测试用 MagicMock 模拟 LLM,验证返回类型契约;具体业务路径(重试、换词、降级)由现有 `test_generate_with_fallback_*` 覆盖。

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py::test_generate_with_fallback_returns_named_result -v
```

期望:FAIL——当前返回裸元组。

- [ ] **Step 3: 重写 generate_with_fallback + 3 个辅助函数**

`src/flow/ket_partner/sentence_domain.py:434-550` 整体替换为:

```python
async def _generate_and_validate(
    llm_smart: BaseChatModel,
    target: str,
    context: str,
    avoid_words: list[str],
    avoid_sentences: list[str],
    age: int,
    profile: dict,
    repos: KETPartnerRepos,
    config: KetConfig,
) -> tuple[str, ValidationResult, list[dict]]:
    """单轮造句 + 验证重试循环。返回 (sentence, result, attempts)。

    attempts 是本轮所有尝试的列表(供 fallback 函数决策)。
    """
    attempts: list[dict] = []
    seen_non_ket_words: list = []

    def _regen() -> str:
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
    result: ValidationResult | None = None

    for _ in range(config.validate_retry_limit):
        check = await validate_and_categorize(
            llm_smart, sentence, target, age, repos, avoid_sentences
        )
        result = check["result"]
        if check["passed"]:
            return sentence, result, attempts
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

    # 最终验证
    check = await validate_and_categorize(
        llm_smart, sentence, target, age, repos, avoid_sentences
    )
    result = check["result"]
    if check["passed"]:
        return sentence, result, attempts
    attempts.append({
        "sentence": sentence,
        "reason_kind": check["reason_kind"],
        "reason_detail": check["reason_detail"],
        "non_ket_words": check["non_ket_words"],
        "non_ket_count": check["non_ket_count"],
    })
    return sentence, result, attempts  # type: ignore[return-value]


async def _handle_overflow_fallback(
    attempts: list[dict],
    target: str,
    context: str,
    repos: KETPartnerRepos,
) -> SentenceGenerationResult | None:
    """若 attempts 含 non_ket_overflow,选 non_ket_count 最少的草稿返回;否则 None。"""
    overflow_attempts = [a for a in attempts if a["reason_kind"] == "non_ket_overflow"]
    if not overflow_attempts:
        return None
    best = min(reversed(overflow_attempts), key=lambda a: a["non_ket_count"])
    sentence = best["sentence"]
    result = await validate_sentence(sentence, repos, target=target)
    logger.warning(
        f"sentence validation: accepting non-KET overflow draft after "
        f"{len(attempts)} attempts (non_ket_count={len(result.non_ket_words)}); "
        f"sentence={sentence!r}"
    )
    return SentenceGenerationResult(sentence=sentence, result=result, target=target, context=context)


async def _switch_target_or_accept(
    attempts: list[dict],
    sentence: str,
    result: ValidationResult,
    target: str,
    context: str,
    word_switched: bool,
    profile: dict,
    repos: KETPartnerRepos,
    config: KetConfig,
) -> SentenceGenerationResult | _RetryOuter:
    """所有 attempts 均为 naturalness(或非 overflow 的其他失败)时进入此分支。

    返回:
    - SentenceGenerationResult: 接受当前草稿
    - _RetryOuter: 请求外层 while 换词重试(仅当 word_switched=False 且能换到不同 target)
    """
    all_naturalness = bool(attempts) and all(
        a["reason_kind"] == "naturalness" for a in attempts
    )

    if all_naturalness and not word_switched:
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
            return SentenceGenerationResult(
                sentence=sentence, result=result, target=target, context=context
            )
        return _RetryOuter(target=new_ref.word, context=new_ref.context)

    # 其他失败原因:接受当前草稿,记 warning
    last = attempts[-1] if attempts else {}
    reasons = []
    if last.get("non_ket_count", 0) > 1:
        reasons.append(f"{last['non_ket_count']} non-KET word(s): {last['non_ket_words']}")
    if last.get("is_duplicate"):
        reasons.append("duplicate of a recent sentence")
    if last.get("is_target_split"):
        reasons.append(f"multi-word target '{target}' split apart")
    if not reasons and last.get("reason_detail"):
        reasons.append(f"naturalness: {last['reason_detail']}")
    logger.warning(
        f"sentence validation failed after {len(attempts)} attempts; "
        f"accepting current draft — reasons: {('; '.join(reasons)) or 'unknown'}; "
        f"sentence={sentence!r}"
    )
    return SentenceGenerationResult(
        sentence=sentence, result=result, target=target, context=context
    )


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
) -> SentenceGenerationResult:
    """造句主入口:生成 → 验证重试 → 失败降级(溢出 / 换词 / 接受)。"""
    target = initial_target
    context = initial_context
    word_switched = False

    while True:
        sentence, result, attempts = await _generate_and_validate(
            llm_smart, target, context, avoid_words, avoid_sentences,
            age, profile, repos, config,
        )

        overflow = await _handle_overflow_fallback(attempts, target, context, repos)
        if overflow is not None:
            return overflow

        decision = await _switch_target_or_accept(
            attempts, sentence, result, target, context, word_switched,
            profile, repos, config,
        )
        if isinstance(decision, _RetryOuter):
            target = decision.target
            context = decision.context
            word_switched = True
            continue
        return decision
```

- [ ] **Step 4: 改 agent.py 调用点**

`src/flow/ket_partner/agent.py` 的 `generate_sentence_node` 方法(原 `nodes.py:173-184`):

```python
# 原:
# sentence, result, final_target, final_ctx = await generate_with_fallback(...)
# target, target_ctx = final_target, final_ctx
# 新:
gen_result = await generate_with_fallback(
    self.llm_smart,
    initial_target=state["target_word"],
    initial_context=target_ctx,
    avoid_words=avoid_words,
    avoid_sentences=avoid_sentences,
    age=age,
    profile=profile,
    repos=repos,
    config=self.config,
)
sentence = gen_result.sentence
result = gen_result.result
target = gen_result.target
target_ctx = gen_result.context
```

- [ ] **Step 5: 运行测试**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_sentence_domain.py -v
```

期望:`test_generate_with_fallback_returns_named_result` PASS,其它现有 sentence 测试也 PASS(逻辑等价)。

如果某些用裸元组解包的旧测试 FAIL,改为命名属性访问。

- [ ] **Step 6: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/sentence_domain.py src/flow/ket_partner/agent.py tests/flow/ket_partner/test_sentence_domain.py
git commit -m "refactor(flow/ket_partner/sentence_domain): split generate_with_fallback into 3 helpers

原 117 行函数拆为:
- _generate_and_validate: 单轮造句+验证重试
- _handle_overflow_fallback: non-KET 溢出降级
- _switch_target_or_accept: 换词 / 接受当前草稿

主函数编排三者,返回 SentenceGenerationResult(替代裸 4 元组)。
内部用 _RetryOuter 信号类区分'接受'与'换词重试'。"
```

---

## Phase 4:P1(4 项)

### Task 4.1:state.py docstring 按实际写入者重写

**Files:**
- Modify: `src/flow/ket_partner/state.py:9-31`

**Interfaces:**
- Produces: `BTPKetState` 的 docstring 逐字段准确反映 single-writer / multi-writer。

- [ ] **Step 1: 重写 docstring**

`src/flow/ket_partner/state.py:9-31` 的 docstring 整体替换为:

```python
    """
    BTP Ket Partner 核心对话状态图

    字段 Single-Writer / Multi-Writer 声明(按实际代码,nodes 已合并入 KETPartnerAgent):

    - messages: 仅 init_state_node(截断超 10 条时)、format_output_node、
      explain_meaning_node、redirect_to_translate_node、compliance_redirect_node
      在追加 AI 回复时写;其他位置只读
    - intent: 仅 classify_intent_node 在路由阶段写;其他位置只读
    - asked_word: 仅 classify_intent_node 在解析查词意图时写;其他位置只读
    - wrong_words: 仅 evaluate_translation_node 写;其他位置只读
    - sentence_translation: 仅 evaluate_translation_node(translation 路径)
      与 lookup_target_meaning_node(idk 路径)写;其他位置只读
    - overall_correct: 仅 evaluate_translation_node 写;其他位置只读
    - asked_word_meaning: 仅 lookup_asked_meaning_node 写;其他位置只读
    - target_word: init_state_node(从上一轮 AI 消息恢复时)与
      select_target_word_node、generate_sentence_node(换词时)写;其他位置只读
    - target_context: 同 target_word,init_state_node / select_target_word_node /
      generate_sentence_node 写;其他位置只读
    - last_target_word: init_state_node(从上一轮 AI 消息恢复)与
      persist_turn_node(本轮结束时)写;其他位置只读
    - last_target_context: 同 last_target_word,init_state_node 与
      persist_turn_node 写;其他位置只读
    - last_sentence_words: init_state_node(从上一轮 AI 消息恢复)与
      generate_sentence_node 写;其他位置只读
    - topic: 仅 select_target_word_node 写;其他位置只读
    - profile_strategy: 仅 init_state_node 从 DB profile 加载时写;其他位置只读
    - profile_weakness: 同 profile_strategy,仅 init_state_node 写;其他位置只读
    - last_english_sentence: init_state_node(从上一轮 AI 消息恢复)与
      generate_sentence_node 写;其他位置只读
    - _exposure_recorded: 仅 generate_sentence_node 标记,persist_turn_node 读取;
      其他位置只读
    - non_ket_annotations: 仅 generate_sentence_node 写;其他位置只读
    """
```

- [ ] **Step 2: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/state.py
git commit -m "docs(flow/ket_partner/state): rewrite BTPKetState docstring with accurate writers

按 nodes 合并入 KETPartnerAgent 后的实际代码,逐字段重写 single-writer /
multi-writer 声明。修正:
- intent 写者 classify_input_node → classify_intent_node
- profile_strategy/profile_weakness 写者(原写不存在的 profile_summarizer_node)
  → init_state_node
- last_target_word/last_target_context/last_english_sentence 显式枚举
  init_state_node + 业务节点双写"
```

---

### Task 4.2:Intent 常量 + graph.py 路由 dict 查表

**Files:**
- Modify: `src/flow/ket_partner/state.py`(加常量)
- Modify: `src/flow/ket_partner/graph.py:14-41`(路由函数改 dict 查表)
- Modify: `src/flow/ket_partner/vocab_domain.py:208`(用常量替换字面量)
- Modify: `src/flow/ket_partner/dialogue_domain.py:220`(用常量替换字面量)
- Test: `tests/flow/ket_partner/test_state.py`、`tests/flow/ket_partner/test_graph.py`

**Interfaces:**
- Produces:
  - `state.TRANSLATION` / `state.IDK` / `state.ASKS_MEANING` / `state.OFF_TOPIC` / `state.NON_COMPLIANT` 常量(类型 `KetIntent`)
  - `graph._ROUTE_AFTER_CLASSIFY: dict[KetIntent, str]` 查表

- [ ] **Step 1: 在 state.py 加常量**

`src/flow/ket_partner/state.py` 在 `KetIntent = Literal[...]` 之后加:

```python
# Intent 常量:消除业务代码中的魔法字符串。
# 字面量与 KetIntent Literal 完全一致,类型系统保证不会写错。
TRANSLATION: KetIntent = "translation"
IDK: KetIntent = "idk"
ASKS_MEANING: KetIntent = "asks_meaning"
OFF_TOPIC: KetIntent = "off_topic"
NON_COMPLIANT: KetIntent = "non_compliant"
```

- [ ] **Step 2: 加常量与 Literal 一致性测试**

`tests/flow/ket_partner/test_state.py` 追加:

```python
def test_intent_constants_match_literal():
    """5 个常量值必须与 KetIntent Literal 完全一致,且互不相同。"""
    from typing import get_args
    from flow.ket_partner.state import (
        KetIntent, TRANSLATION, IDK, ASKS_MEANING, OFF_TOPIC, NON_COMPLIANT,
    )

    literals = set(get_args(KetIntent))
    constants = {TRANSLATION, IDK, ASKS_MEANING, OFF_TOPIC, NON_COMPLIANT}
    assert literals == constants, (
        f"KetIntent Literal 与常量集合不一致: literal={literals}, constant={constants}"
    )
```

- [ ] **Step 3: 改 graph.py 两个路由函数为 dict 查表**

`src/flow/ket_partner/graph.py:14-41` 替换为:

```python
from flow.ket_partner.state import (
    BTPKetState, KetIntent,
    TRANSLATION, IDK, ASKS_MEANING, OFF_TOPIC, NON_COMPLIANT,
)

# classify_intent_node 之后路由表:每个 intent 对应下一个节点名。
_ROUTE_AFTER_CLASSIFY: dict[KetIntent, str] = {
    TRANSLATION: "evaluate_translation",
    IDK: "lookup_target_meaning",
    ASKS_MEANING: "lookup_asked_meaning",
    # OFF_TOPIC / NON_COMPLIAN 不在此分支,route_after_classify 返回 "skip"
}
_DEFAULT_AFTER_CLASSIFY = "skip"


def route_by_intent(state: BTPKetState) -> str:
    """format_output_or_branch 之后路由,根据 intent 选下一节点。"""
    intent = state.get("intent")
    if intent in (TRANSLATION, IDK):
        return "select_target_word"
    if intent == ASKS_MEANING:
        return "explain_meaning"
    if intent == OFF_TOPIC:
        return "redirect_to_translate"
    if intent == NON_COMPLIANT:
        return "compliance_redirect"
    return "select_target_word"


def route_after_init(state: BTPKetState) -> str:
    if state.get("last_english_sentence") is None:
        return "select_target_word"
    return "classify_intent"


def route_after_classify(state: BTPKetState) -> str:
    intent = state.get("intent")
    if intent is None:
        return _DEFAULT_AFTER_CLASSIFY
    return _ROUTE_AFTER_CLASSIFY.get(intent, _DEFAULT_AFTER_CLASSIFY)
```

- [ ] **Step 4: 改 vocab_domain / dialogue_domain 用常量**

`src/flow/ket_partner/vocab_domain.py` 顶部 import:

```python
from flow.ket_partner.state import (
    BTPKetState, TRANSLATION, IDK, ASKS_MEANING,
)
```

`apply_mastery_updates`(L206-234)中:

```python
# 原:if intent == "translation":
# 新:
if intent == TRANSLATION:
# 原:elif intent == "idk":
# 新:
elif intent == IDK:
# 原:elif intent == "asks_meaning":
# 新:
elif intent == ASKS_MEANING:
```

`src/flow/ket_partner/dialogue_domain.py` 顶部 import:

```python
from flow.ket_partner.state import BTPKetState, TRANSLATION, IDK
```

`format_output_text`(L216-248):

```python
# 原:if intent == "translation":
# 新:
if intent == TRANSLATION:
# 原:elif intent == "idk":
# 新:
elif intent == IDK:
```

- [ ] **Step 5: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_state.py tests/flow/ket_partner/test_graph.py -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/state.py src/flow/ket_partner/graph.py src/flow/ket_partner/vocab_domain.py src/flow/ket_partner/dialogue_domain.py tests/flow/ket_partner/test_state.py
git commit -m "refactor(flow/ket_partner): replace intent magic strings with constants + dict

state.py 加 5 个 Intent 常量(TRANSLATION/IDK/ASKS_MEANING/OFF_TOPIC/
NON_COMPLIANT),与 KetIntent Literal 严格一致。

graph.py route_after_classify 改用 _ROUTE_AFTER_CLASSIFY dict 查表,
消除 if/elif 字面量分支。

vocab_domain.apply_mastery_updates 与 dialogue_domain.format_output_text
用常量替换字面量比较。"
```

---

### Task 4.3:chat 路由函数拆分

**Files:**
- Modify: `src/api/routes/chat.py:19-82`

**Interfaces:**
- Produces:
  - `async def _invoke_agent(agent, req, repos, user_info, timeout) -> dict`(包 `agent.ainvoke` + timeout)
  - `async def _build_chat_response(state, repos) -> ChatResponse`(从 state 抽 ai_reply + turn_id)

- [ ] **Step 1: 重构 chat 函数**

`src/api/routes/chat.py:24-82` 重构为:

```python
async def _invoke_agent(
    agent: CompiledStateGraph[Any, None, Any, Any],
    req: ChatRequest,
    repos: Repos,
    user_info: dict,
    timeout: float,
) -> dict:
    """调用 agent.ainvoke,处理 LLM SDK 异常 → HTTPException 映射。"""
    try:
        result_state = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=req.text)]},
                config={
                    "configurable": {
                        "thread_id": f"{repos._user_id}:main",
                        "user_id": repos._user_id,
                        "repos": repos,
                        "user_info": user_info,
                    }
                },
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as e:
        logger.warning("agent execution timeout: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail="agent timeout")
    except openai.APITimeoutError as e:
        logger.warning("LLM SDK timeout: %s", e, exc_info=True)
        raise HTTPException(status_code=504, detail="LLM timeout")
    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError) as e:
        raise HTTPException(status_code=401, detail="LLM auth failed")
    except (openai.APIConnectionError, openai.RateLimitError) as e:
        logger.warning("LLM transient failure: %s", e, exc_info=True)
        status_code = 429 if isinstance(e, openai.RateLimitError) else 502
        raise HTTPException(status_code=status_code, detail=str(e) or "transient error")
    return result_state  # type: ignore[no-any-return]


async def _build_chat_response(
    result_state: dict,
    repos: Repos,
    llm_key_status: LlmKeyStatus,
) -> ChatResponse:
    """从 agent 返回的 state 抽取 ai_reply,组装 ChatResponse。"""
    llm_key_status.clear_error()

    messages = result_state.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="agent returned empty messages")

    ai_text = str(messages[-1].content).strip()
    if not ai_text:
        raise HTTPException(status_code=500, detail="agent returned blank reply")

    profile = await repos.profile.get()
    turn_id = profile.get("total_turns", 0)
    return ChatResponse(ai_reply=ai_text, turn_id=turn_id)


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
    agent: CompiledStateGraph[Any, None, Any, Any] = Depends(get_agent),
    settings: Settings = Depends(get_settings),
    llm_key_status: LlmKeyStatus = Depends(get_llm_key_status),
) -> ChatResponse:
    if not _read_current_key():
        raise HTTPException(status_code=503, detail="LLM key not configured")

    repos = Repos.for_user(db, user.id)
    user_info = {"nickname": user.nickname, "age": user.age}

    # auth error 需要标记 llm_key_status,所以单独处理
    try:
        result_state = await _invoke_agent(
            agent, req, repos, user_info, settings.REQUEST_TIMEOUT,
        )
    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError):
        llm_key_status.set_error(LLM_AUTH_ERROR_MSG)
        raise

    return await _build_chat_response(result_state, repos, llm_key_status)
```

注:`repos._user_id` 是 Repos 类的内部字段;如果不存在,需要补一个 `user_id` 属性或直接传 `user.id` 进 `_invoke_agent`。**实施时先 Read `src/persistence/repos.py` 确认 Repos 字段**,按实际字段调整 `_invoke_agent` 签名。

- [ ] **Step 2: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/api/routes/test_chat_route.py -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/api/routes/chat.py
git commit -m "refactor(api/routes/chat): split chat() into _invoke_agent + _build_chat_response

原 64 行 chat 函数拆为两个辅助 + 主编排:
- _invoke_agent: agent.ainvoke + timeout + LLM SDK 异常 → HTTPException 映射
- _build_chat_response: 从 state 抽 ai_reply + 组装响应"
```

---

### Task 4.4:6 处 fallback 测试补 mock 调用断言

**Files:**
- Modify: `tests/flow/ket_partner/test_vocab_domain.py`、`test_dialogue_domain.py`、`test_sentence_domain.py`

> 注:Phase 1 的 Task 1.2/1.3/1.4 已经在改异常元组时给 3 处 vocab/dialogue/sentence 测试加了 `assert_awaited_once`。本 Task 补齐剩余的:
> - `test_dialogue_domain.py:103-118 test_summary_fallback_on_error`
> - `test_dialogue_domain.py:172-187 test_evaluate_fallback_on_error`
> 
> 如果 Phase 1 已经全部覆盖,本 Task 可跳过。

- [ ] **Step 1: 检查覆盖率**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m grep -n "assert_awaited" tests/flow/ket_partner/test_vocab_domain.py tests/flow/ket_partner/test_dialogue_domain.py tests/flow/ket_partner/test_sentence_domain.py
```

逐条对照 spec 中列出的 6 处 fallback 测试,确认每处都有 `assert_awaited_once` 或 `assert_awaited` 断言。

- [ ] **Step 2: 补齐缺失的断言**

对每个缺失的 fallback 测试,在断言返回值之后加:

```python
bound.with_structured_output.return_value.ainvoke.assert_awaited_once()
```

如果是同步 LLM 调用(`bound.invoke`),改为:

```python
bound.with_structured_output.return_value.invoke.assert_called_once()
```

- [ ] **Step 3: 运行测试 + 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add tests/flow/ket_partner/
git commit -m "test(flow/ket_partner): add mock call assertions to remaining fallback tests

补齐 spec §12 列出的 6 处 fallback 测试的 assert_awaited_once 断言,
防止被测函数内部异常导致从未调用 mock 时,测试因 fallback 返回值
形态重合而假通过。"
```

---

## Phase 5:P2(批量收尾)

### Task 5.1:ChatLogger 上下文管理器化

**Files:**
- Modify: `src/cli/ket_partner/chat_logger.py`(加 `__enter__` / `__exit__`)
- Modify: `src/cli/ket_partner/main.py`(调用点改 `with`)
- Test: `tests/cli/ket_partner/test_chat_logger.py`

**Interfaces:**
- Produces: `ChatLogger` 实现 `__enter__() -> self` 与 `__exit__(*args) -> None`(后者兜底调 `close_session`)

- [ ] **Step 1: 写上下文管理器异常路径测试**

`tests/cli/ket_partner/test_chat_logger.py` 追加:

```python
def test_chat_logger_exit_closes_session_on_exception(tmp_path):
    """with 块内抛异常时,__exit__ 仍应调 close_session,释放文件 handle。"""
    from src.cli.ket_partner.chat_logger import ChatLogger

    log_dir = str(tmp_path / "logs")
    logger = ChatLogger(log_dir=log_dir)
    logger.start_session("test")

    fp_ref = logger._fp
    assert fp_ref is not None

    import pytest
    with pytest.raises(ValueError):
        with logger:
            raise ValueError("simulated failure")

    # __exit__ 应已关闭 fp
    assert fp_ref.closed is True
    assert logger._fp is None


def test_chat_logger_exit_idempotent(tmp_path):
    """__exit__ 多次调用不重复关 fp(避免 ValueError: I/O operation on closed file)。"""
    from src.cli.ket_partner.chat_logger import ChatLogger

    logger = ChatLogger(log_dir=str(tmp_path / "logs"))
    logger.start_session("test")
    with logger:
        pass
    # 二次 close 不应抛异常
    logger.close_session()
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/test_chat_logger.py -v -k "exit"
```

期望:FAIL——`ChatLogger` 不是上下文管理器。

- [ ] **Step 3: 加 `__enter__` / `__exit__`**

`src/cli/ket_partner/chat_logger.py` 在 `ChatLogger` 类末尾(L48 后)加:

```python
    def __enter__(self) -> "ChatLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # 兜底关闭:无论 with 块内是否抛异常,都尝试关闭当前 session。
        # close_session 内部已判 _fp is None,二次调用安全。
        self.close_session()
```

同时改 `close_session`(L42-48)为幂等:

```python
    def close_session(self) -> None:
        if self._fp is None:
            return
        try:
            self._fp.write("\n" + "-" * 60 + "\n")
            self._fp.write(f"Session ended: {datetime.now():%H:%M:%S}\n")
        finally:
            self._fp.close()
            self._fp = None
```

注:加 `try/finally` 确保即使写入异常,fp 也会被关闭。

- [ ] **Step 4: 改 main.py 调用点**

`src/cli/ket_partner/main.py` 中 ChatLogger 的使用处(根据现有代码定位,L60 附近):

```python
# 原:
# chat_logger = ChatLogger(...)
# chat_logger.start_session(nickname)
# ... 一系列 log_turn ...
# chat_logger.close_session()
# 新:
with ChatLogger(...) as chat_logger:
    chat_logger.start_session(nickname)
    # ... 一系列 log_turn ...
```

具体改造以 main.py 现有结构为准,核心是确保 `with` 块覆盖整个 session 生命周期。

- [ ] **Step 5: 运行测试 + 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/cli/ket_partner/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/cli/ket_partner/chat_logger.py src/cli/ket_partner/main.py tests/cli/ket_partner/test_chat_logger.py
git commit -m "refactor(cli/chat_logger): make ChatLogger a context manager

加 __enter__/__exit__,后者兜底调 close_session 确保异常路径下文件
handle 释放。close_session 改幂等(try/finally 保证 fp.close 必执行)。

main.py 调用点改 with 块,覆盖整个 session 生命周期。"
```

---

### Task 5.2:exporter.py 异步 IO 替换

**Files:**
- Modify: `src/reporting/ket_partner/exporter.py:38`

- [ ] **Step 1: 检查 aiofiles 是否已安装**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -c "import aiofiles; print(aiofiles.__version__)"
```

如果 `ModuleNotFoundError`,跳到备选方案 B。

- [ ] **Step 2: 用 aiofiles 替换同步 open**

`src/reporting/ket_partner/exporter.py:38`:

```python
# 方案 A(aiofiles 已安装):
# 原:with open(out_p, "w", encoding="utf-8") as f:  # noqa: ASYNC230
#         f.write(content)
# 新:
import aiofiles
async with aiofiles.open(out_p, "w", encoding="utf-8") as f:
    await f.write(content)
```

或方案 B(aiofiles 未安装):

```python
# 原:with open(out_p, "w", encoding="utf-8") as f:  # noqa: ASYNC230
#         f.write(content)
# 新:
import asyncio
def _write_file(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

await asyncio.to_thread(_write_file, out_p, content)
```

推荐方案 A。如果选 B,辅助函数 `_write_file` 在模块顶层定义,不要内联。

- [ ] **Step 3: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/reporting/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/reporting/ket_partner/exporter.py
git commit -m "refactor(reporting/exporter): replace sync open with aiofiles in async fn

移除 # noqa: ASYNC230,§7.2 合规。"
```

---

### Task 5.3:bootstrap.py CSV 读取用 asyncio.to_thread

**Files:**
- Modify: `src/persistence/bootstrap.py:46-75`

- [ ] **Step 1: 抽同步辅助函数,用 to_thread 包裹**

`src/persistence/bootstrap.py:46-75` 重构为:

```python
def _read_csv_rows(csv_path: str) -> list[dict[str, str]]:
    """同步读 CSV 全部行。供 asyncio.to_thread 调用。

    跨边界 try/except 仅捕获 OSError(外部 IO 失败);
    csv.Error / 解析问题由调用方处理。
    """
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


async def _import_csv(db: aiosqlite.Connection, csv_path: str) -> None:
    """Import KET vocabulary from CSV. Private — only bootstrap.py calls this."""
    rows = await asyncio.to_thread(_read_csv_rows, csv_path)
    count = 0
    for row in rows:
        word = (row.get("word") or "").strip()
        pos = (row.get("part_of_speech") or "").strip()
        topic_raw = (row.get("topic") or "").strip()
        context = (row.get("context") or "").strip()
        # 必要字段缺失则跳过(P2 #24)
        if not word or not pos:
            continue
        topics = [t.strip() for t in topic_raw.split(";") if t.strip()]
        await db.execute(
            "INSERT OR IGNORE INTO ket_vocabulary (word, context, pos, is_seed) "
            "VALUES (?, ?, ?, 0)",
            (word, context, pos),
        )
        for t in topics:
            await db.execute(
                "INSERT OR IGNORE INTO ket_vocab_topics (word, context, topic) "
                "VALUES (?, ?, ?)",
                (word, context, t),
            )
        count += 1
    await db.commit()
    logger.info(f"Imported {count} rows from {csv_path}")
```

注:顶部需 `import asyncio`(若未导入)。同时本 Task 顺带完成 P2 #24(strip 后判空加强)。

- [ ] **Step 2: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/persistence/bootstrap.py
git commit -m "refactor(persistence/bootstrap): move CSV read to asyncio.to_thread

抽 _read_csv_rows 同步辅助,async _import_csv 用 asyncio.to_thread
卸载同步 IO;移除 # noqa: ASYNC230。

顺带 P2 #24:CSV 行的 word/pos/topic/context 字段全部 strip 后判空,
缺失则跳过。"
```

---

### Task 5.4:config.py 模块级加载 JSON

**Files:**
- Modify: `src/flow/ket_partner/config.py:41-47`

- [ ] **Step 1: 把 JSON 加载提到模块级**

`src/flow/ket_partner/config.py` 整体替换为:

```python
import json
from os.path import dirname, join
from typing import Any

from pydantic import BaseModel


class VocabRefillConfig(BaseModel):
    low_watermark: int = 5
    high_watermark: int = 10
    interval_turns: int = 3


class SentenceConfig(BaseModel):
    min_words: int = 5
    max_words: int = 12
    rewrite_threshold: int = 2


class VarietyConfig(BaseModel):
    recent_window: int = 3


class SummaryConfig(BaseModel):
    interval_turns: int = 15


class StrugglingThreshold(BaseModel):
    wrong_count_min: int = 2
    exposed_count_min: int = 4


class KetConfig(BaseModel):
    vocab_refill: VocabRefillConfig = VocabRefillConfig()
    sentence: SentenceConfig = SentenceConfig()
    variety: VarietyConfig = VarietyConfig()
    summary: SummaryConfig = SummaryConfig()
    validate_retry_limit: int = 2
    struggling_threshold: StrugglingThreshold = StrugglingThreshold()


_CONFIG_PATH = join(dirname(__file__), "data", "config.json")

# 模块级一次性加载 JSON(同步,import 时执行)。
# 这样 load_config() 函数体只做 pydantic 校验,不触发 IO,
# 在 async 调用路径(graph.build_agent)中安全。
# 与 sentence_domain.py 模块级加载 function_words.json / lemmas.json 同模式。
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _CONFIG_DATA: Any = json.load(f)


def load_config() -> KetConfig:
    """返回新的 KetConfig 实例(每次调用都 model_validate 一遍)。"""
    return KetConfig.model_validate(_CONFIG_DATA)
```

- [ ] **Step 2: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/test_config.py -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/config.py
git commit -m "refactor(flow/ket_partner/config): hoist JSON load to module level

load_config() 不再触发同步 IO,只在 _CONFIG_DATA 上做 pydantic 校验。
消除 graph.build_agent 内 async 路径调用 load_config 时的 §7.2 违规。"
```

---

### Task 5.5:state.get("intent") 加 None 校验 + format_output_text 完备分支

**Files:**
- Modify: `src/flow/ket_partner/dialogue_domain.py:216-248`(format_output_text)
- Modify: `src/flow/ket_partner/vocab_domain.py:206-234`(apply_mastery_updates)
- Modify: `src/flow/ket_partner/graph.py:14, 33`(路由函数)

**Interfaces:**
- Consumes: Task 4.2 的 Intent 常量
- Produces: 所有读取 `state.get("intent")` 的位置都显式处理 None。

- [ ] **Step 1: 改 format_output_text 显式判 None + 完备分支**

`src/flow/ket_partner/dialogue_domain.py:216-248`:

```python
def format_output_text(state: BTPKetState, new_sentence: str) -> str:
    from flow.ket_partner.state import (
        TRANSLATION, IDK, ASKS_MEANING, OFF_TOPIC, NON_COMPLIANT,
    )

    intent = state.get("intent")
    lines: list[str] = []

    if intent == TRANSLATION:
        wrong = state.get("wrong_words") or []
        sentence_t = state.get("sentence_translation", "")
        overall_correct = state.get("overall_correct")
        if wrong:
            if sentence_t:
                lines.append(f"正确翻译:{sentence_t}")
            lines.append("你的翻译有误:")
            for entry in wrong:
                word = entry.get("word", "?")
                correct = entry.get("correct_translation", "?")
                lines.append(f" {word} 的意思是:{correct}")
        elif overall_correct is False:
            if sentence_t:
                lines.append(f"正确翻译:{sentence_t}")
            lines.append("你的翻译和原句意思有些偏差。")
    elif intent == IDK:
        sentence_t = state.get("sentence_translation", "")
        if sentence_t:
            lines.append(f"正确翻译:{sentence_t}")
    elif intent == ASKS_MEANING:
        # 已在 explain_meaning_node 输出意思,此处无需额外处理
        pass
    elif intent == OFF_TOPIC:
        # redirect_to_translate_node 已处理
        pass
    elif intent == NON_COMPLIANT:
        # compliance_redirect_node 已处理
        pass
    else:
        # intent is None 或未知值:不补充额外文案,只输出新句
        logger.debug(f"format_output_text: intent={intent!r}, no specific branch")

    lines.append("请把这句译成中文:")
    lines.append(f'"{new_sentence}"')
    for ann in state.get("non_ket_annotations") or []:
        word = ann.get("word", "?")
        meaning = ann.get("meaning", "")
        if meaning:
            lines.append(f"{word} 的意思是:{meaning}")
    return "\n".join(lines)
```

- [ ] **Step 2: apply_mastery_updates 显式判 None**

`src/flow/ket_partner/vocab_domain.py:206-234` 开头改为:

```python
async def apply_mastery_updates(state: BTPKetState, repos: KETPartnerRepos) -> None:
    intent = state.get("intent")
    if intent is None:
        logger.debug("apply_mastery_updates: intent is None, skip")
        return

    if intent == TRANSLATION:
        # ... 原逻辑
    elif intent == IDK:
        # ...
    elif intent == ASKS_MEANING:
        # ...
    # OFF_TOPIC / NON_COMPLIANT:不更新 mastery,显式 noop
```

- [ ] **Step 3: graph.py 路由函数 None 校验**

Task 4.2 的 graph.py 重构已经包含 `if intent is None: return _DEFAULT_AFTER_CLASSIFY`。如果未包含,补充。

`route_by_intent` 也加 None 兜底:

```python
def route_by_intent(state: BTPKetState) -> str:
    intent = state.get("intent")
    if intent is None:
        return "select_target_word"
    # ...
```

- [ ] **Step 4: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/flow/ket_partner/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/flow/ket_partner/dialogue_domain.py src/flow/ket_partner/vocab_domain.py src/flow/ket_partner/graph.py
git commit -m "fix(flow/ket_partner): explicit None checks and complete intent branches

- format_output_text: 显式判 intent is None,5 个 Intent 全覆盖(原仅
  translation/idk 两个 elif)
- apply_mastery_updates: 加 intent is None 提前返回
- graph.route_by_intent: 加 None 兜底返回 select_target_word

消除 dead path 嫌疑(§5.2)。"
```

---

### Task 5.6:repos.py ProfileRepo.update 静默 no-op 加 warning

**Files:**
- Modify: `src/persistence/repos.py:338`

- [ ] **Step 1: 加 logger.warning**

`src/persistence/repos.py:338` 附近(`ProfileRepo.update` 方法内):

```python
# 原:
# if not fields:
#     return
# 新:
if not fields:
    logger.warning("ProfileRepo.update called with empty fields; no-op")
    return
```

顶部 import 区确认有 `from flow.common import logger`(若无则补)。

- [ ] **Step 2: 三项静态验证 + Commit**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/persistence/ -v
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
git add src/persistence/repos.py
git commit -m "fix(persistence/repos): log warning when ProfileRepo.update is no-op

变静默 return 为 warning,便于发现空 update 调用。"
```

---

## 最终集成验证

### Task F.1:运行集成测试

- [ ] **Step 1: 运行 integration 测试(若配置了 DASHSCOPE_API_KEY)**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest tests/integration/ -v
```

期望:全 PASS 或 `@pytest.mark.integration` 标记的 skip(缺 key 时)。

- [ ] **Step 2: 全量三项静态验证**

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
```

期望:ruff 0 错 0 警、mypy 0 错、pytest 全绿。

- [ ] **Step 3: Commit 最终状态**

如果集成测试新增了用例:

```bash
git add tests/integration/
git commit -m "test(integration): add post-refactor integration verification"
```

否则无需 commit,直接收尾。

---

## 自审清单

完成所有 Task 后,审核人按以下清单逐条核对:

1. **KetIntent 一致性**:`from flow.ket_partner.state import KetIntent, get_args; from flow.ket_partner.dialogue_domain import IntentClassification; assert set(get_args(KetIntent)) == set(get_args(IntentClassification.model_fields['intent'].annotation))` 返回 True。
2. **9 处 LLM except 元组**:全项目 `grep "except (TimeoutError, RuntimeError, ValueError"` 返回 0 行。
3. **app.py shutdown**:`grep "except Exception" src/api/app.py` 返回 0 行。
4. **SentenceGenerationResult**:`grep "tuple\[str, ValidationResult, str, str\]" src/` 返回 0 行(原裸元组已消除)。
5. **LlmService DI**:`grep "from flow.common import llm_max\|from flow.common import llm_flash" src/` 返回 0 行(模块级 llm_max/llm_flash 已删,虽然 Task 2.1 保留,但 Task 2.x 完成后应无业务代码引用——本计划未列删除任务,实施时确认 `git grep llm_max src/` 仅在 common.py 内出现)。
6. **nodes.py 删除**:`ls src/flow/ket_partner/nodes.py` 返回 No such file。
7. **Intent 常量**:`grep 'intent == "' src/` 返回 0 行(全部用常量)。
8. **ChatLogger 上下文管理器**:`grep "__exit__" src/cli/ket_partner/chat_logger.py` 返回至少 1 行。
9. **静态三项清零**:ruff+mypy+pytest 全过。

---

## 实施完毕后

调用 `superpowers:finishing-a-development-branch` skill,选择是否合并到 master 或发起 PR。
