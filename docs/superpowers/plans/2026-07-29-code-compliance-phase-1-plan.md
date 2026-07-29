# 第一阶段：静态类型与门禁打底 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `src/flow/agent.py` 与 `src/flow/english_training_partner.py` 中的全部 16 处 mypy 静态类型错误，达成静态门禁 (ruff + mypy) 零报错通关。

**Architecture:** 严格按照第一阶段修改规范，针对定位出的 9 组具体的类型隐患进行精确修补与类型收窄，保持运行时逻辑行为完全不变。

**Tech Stack:** Python 3.10+, Mypy, Ruff, Pytest, LangChain Core Messages.

## Global Constraints

- 不改变任何业务逻辑的运行时行为。
- 禁止添加任何 `# type: ignore`。
- 修改后必须通过 `ruff check .` 与 `mypy src` 零报错检查。
- 所有单元测试与集成测试必须通过。

---

### Task 1: 修复 `src/flow/agent.py` 中的 10 处类型错误

**Files:**
- Modify: `src/flow/agent.py:43-218`
- Test: `tests/ket_partner/test_nodes.py`

**Interfaces:**
- Consumes: `langchain_core.messages.BaseMessage`, `langchain_core.messages.AIMessage`, `langchain_core.messages.ToolCall`
- Produces: Type-safe `Autonomous` and `get_tool_call_signature` functions passing Mypy strict check

- [ ] **Step 1: 验证当前 mypy 报错输出**

Run: `mypy src/flow/agent.py`  
Expected: 10 errors found in `src/flow/agent.py`

- [ ] **Step 2: 修复 `get_tool_call_signature` 入参类型注解 (L43-45)**

In `src/flow/agent.py`:
```python
# 导入 Sequence与 Any
from typing import (
    Any,
    Literal,
    Sequence,
    TypedDict,
)
from langchain_core.messages import ToolCall  # 导入 ToolCall 类型

def get_tool_call_signature(
    tool_calls: Sequence[ToolCall | dict[str, Any]],
) -> str:
    extracted = []
    for tc in tool_calls:
        # ToolCall 既支持 TypedDict 访问，也兼容字典索引
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        args_str = dumps(args, sort_keys=True)
        name = tc["name"] if isinstance(tc, dict) else tc.get("name")
        extracted.append(f"{name}:{args_str}")
    return ";".join(sorted(extracted))
```

- [ ] **Step 3: 修复 `self.hools` 属性注解与 `self.system_message` 类型声明 (L68-74)**

In `src/flow/agent.py`:
```python
        self.system_message: list[BaseMessage] = [
            SystemMessage(content=prompt)
        ]
        self.tools = {
            tool.name: tool for tool in tools
        }
        self.hools: dict[str, Any] = {}
```

- [ ] **Step 4: 在 `tool` 方法中加入 `AIMessage` 类型收窄 (L133-136)**

In `src/flow/agent.py`:
```python
        last_msg = current_messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                ...
```

- [ ] **Step 5: 修复 `condition` 方法中 `Literal` 字面量参数 (L177)**

In `src/flow/agent.py`:
```python
    async def condition(
        self, state: dict
    ) -> Literal["tool", "__end__"]:
```

- [ ] **Step 6: 验证 `src/flow/agent.py` mypy 检查**

Run: `mypy src/flow/agent.py`  
Expected: `Success: no issues found in 1 source file`

- [ ] **Step 7: 提交代码**

```bash
git add src/flow/agent.py
git commit -m "fix(flow/agent): resolve 10 mypy type errors in agent.py"
```

---

### Task 2: 修复 `src/flow/english_training_partner.py` 中的 6 处类型错误

**Files:**
- Modify: `src/flow/english_training_partner.py:518-1655`

**Interfaces:**
- Consumes: `ErrorKeyOutput`, `BaseMessage`, `SystemMessage`
- Produces: Type-safe `english_training_partner.py` passing Mypy strict check

- [ ] **Step 1: 验证当前 mypy 报错输出**

Run: `mypy src/flow/english_training_partner.py`  
Expected: 6 errors found in `src/flow/english_training_partner.py`

- [ ] **Step 2: 修复 `chat_messages` 初始化类型声明 (L518)**

In `src/flow/english_training_partner.py`:
```python
        chat_messages: list[BaseMessage] = [
            SystemMessage(content="你是一个智能助手，能够友好地与用户对话。你生成的对话不能包含emoji。")
        ]
```

- [ ] **Step 3: 修复 `categories` 字典类型声明 (L906)**

In `src/flow/english_training_partner.py`:
```python
        if classification == "list_scenes":
            categories: dict[str, list[str]] = {}
            for s in self.scenes:
```

- [ ] **Step 4: 修复 `result` 变量类型重用与 `error_key` 访问 (L1217-1218)**

In `src/flow/english_training_partner.py`:
```python
                ek_result = await structured_llm.ainvoke(
                    [
                        SystemMessage(...),
                        HumanMessage(...)
                    ]
                )
                if isinstance(ek_result, ErrorKeyOutput):
                    error_key = ek_result.error_key
```

- [ ] **Step 5: 修复 `test_cases` 字典声明 (L1655)**

In `src/flow/english_training_partner.py`:
```python
    test_cases: dict[str, dict[str, Any]] = {"人工交互": {}}
```

- [ ] **Step 6: 验证 `src/flow/english_training_partner.py` mypy 检查**

Run: `mypy src/flow/english_training_partner.py`  
Expected: `Success: no issues found in 1 source file`

- [ ] **Step 7: 提交代码**

```bash
git add src/flow/english_training_partner.py
git commit -m "fix(flow/english_partner): resolve 6 mypy type errors in english_training_partner.py"
```

---

### Task 3: 全项目第一阶段 Quality Gate 门禁验证

**Files:**
- Audit: All codebase

- [ ] **Step 1: 运行 Ruff 代码规范检查**

Run: `ruff check .`  
Expected: All checks passed!

- [ ] **Step 2: 运行 Mypy 全项目类型检查**

Run: `mypy src`  
Expected: `Success: no issues found in 37 source files`

- [ ] **Step 3: 运行 Pytest 测试**

Run: `pytest`  
Expected: All tests pass successfully.

- [ ] **Step 4: 提交合并门禁通过记录**

```bash
git commit --allow-empty -m "ci: complete phase 1 code compliance quality gate check"
```
