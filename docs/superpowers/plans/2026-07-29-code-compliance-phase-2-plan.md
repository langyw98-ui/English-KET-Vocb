# 第二阶段：`ket_partner` 10 维规范合规治理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 `src/flow/ket_partner/`（20个源码文件）与 `tests/ket_partner/`（20个测试文件）进行异常处理降噪重构、State 字段类型收窄与 Single-Writer 文档补全，以及单元测试 Mock 断言强化。

**Architecture:** 严格按照第二阶段修改规范，清除 7 个节点模块中的 9 处 `BLE001` `# noqa` 压制与泛化异常捕获，并在测试中加入 `call_count`/`await_count` 断言。

**Tech Stack:** Python 3.10+, Mypy, Ruff, Pytest, LangChain Core Messages.

## Global Constraints

- 不改变任何业务逻辑的运行时行为。
- 移除所有 `# noqa: BLE001` 屏蔽，改用具体的 `(TimeoutError, RuntimeError, ValueError)` 显式元组捕获。
- 所有兜底分支保留 `logger.warning(..., exc_info=True)`。
- 所有 Mock 单元测试补全 `await_count` / `assert_awaited*` 断言。
- 必须保持 `ruff check .` 与 `mypy src` 0 报错。

---

### Task 1: 节点模块异常处理纪律改造

**Files:**
- Modify: `src/flow/ket_partner/agent.py:440`
- Modify: `src/flow/ket_partner/input_classifier.py:33`
- Modify: `src/flow/ket_partner/profile_summarizer.py:40`
- Modify: `src/flow/ket_partner/sentence_generator.py:126`
- Modify: `src/flow/ket_partner/sentence_naturalness.py:51`
- Modify: `src/flow/ket_partner/translation_evaluator.py:129`
- Modify: `src/flow/ket_partner/word_meaning_lookup.py:72,113,138`

**Interfaces:**
- Consumes: `logger.warning(..., exc_info=True)`
- Produces: Exception-safe nodes without blind except suppression

- [ ] **Step 1: 替换 `agent.py` L440 泛化异常捕获**

In `src/flow/ket_partner/agent.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"Agent LLM 交互失败: {e}", exc_info=True)
```

- [ ] **Step 2: 替换 `input_classifier.py` L33 泛化异常捕获**

In `src/flow/ket_partner/input_classifier.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"输入分类节点异常: {e}", exc_info=True)
```

- [ ] **Step 3: 替换 `profile_summarizer.py` L40 泛化异常捕获**

In `src/flow/ket_partner/profile_summarizer.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"画像生成节点异常: {e}", exc_info=True)
```

- [ ] **Step 4: 替换 `sentence_generator.py` L126 泛化异常捕获**

In `src/flow/ket_partner/sentence_generator.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"例句生成节点异常: {e}", exc_info=True)
```

- [ ] **Step 5: 替换 `sentence_naturalness.py` L51 泛化异常捕获**

In `src/flow/ket_partner/sentence_naturalness.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"自然度校验节点异常: {e}", exc_info=True)
```

- [ ] **Step 6: 替换 `translation_evaluator.py` L129 泛化异常捕获**

In `src/flow/ket_partner/translation_evaluator.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"翻译评估节点异常: {e}", exc_info=True)
```

- [ ] **Step 7: 替换 `word_meaning_lookup.py` L72, L113, L138 泛化异常捕获**

In `src/flow/ket_partner/word_meaning_lookup.py`:
```python
# 修改前
except Exception as e:  # noqa: BLE001

# 修改后
except (TimeoutError, RuntimeError, ValueError) as e:
    logger.warning(f"词义查询失败: {e}", exc_info=True)
```

- [ ] **Step 8: 运行 Ruff 检查确认无 BLE001**

Run: `ruff check src/flow/ket_partner/`  
Expected: `All checks passed!`

- [ ] **Step 9: 提交代码**

```bash
git add src/flow/ket_partner/*.py
git commit -m "refactor(ket_partner): replace blind exception catches with explicit exception tuples"
```

---

### Task 2: State 字段类型收窄与 Single-Writer Docstring 补全

**Files:**
- Modify: `src/flow/ket_partner/state.py`

- [ ] **Step 1: 引入 `KetIntent` 强类型定义**

In `src/flow/ket_partner/state.py`:
```python
from typing import Literal, TypedDict
from langchain_core.messages import AnyMessage

KetIntent = Literal["translate", "asks_meaning", "idk", "off_topic", "non_compliant"]
```

- [ ] **Step 2: 给 `BTPKetState` 补充 Single-Writer docstring**

In `src/flow/ket_partner/state.py`:
```python
class BTPKetState(TypedDict):
    """
    BTP Ket Partner 核心对话状态图

    字段 Single-Writer 声明：
    - messages: 仅 agent_node 与用户交互层写入追加；其他节点只读
    - intent: 仅 classify_input_node 在路由阶段写入；其他节点只读
    - asked_word: 仅 classify_input_node 在解析查词意图时写入；其他节点只读
    - wrong_words: 仅 evaluate_translation_node 写入；其他节点只读
    - sentence_translation: 仅 evaluate_translation_node 与 lookup_target_meaning_node 写入；其他节点只读
    - overall_correct: 仅 evaluate_translation_node 写入；其他节点只读
    - asked_word_meaning: 仅 lookup_target_meaning_node 写入；其他节点只读
    - target_word: 仅 select_vocab_node 写入；其他节点只读
    - target_context: 仅 select_vocab_node 写入；其他节点只读
    - last_target_word: 仅 persist_turn_node 在轮次结束时写入；其他节点只读
    - last_target_context: 仅 persist_turn_node 在轮次结束时写入；其他节点只读
    - last_sentence_words: 仅 generate_sentence_node 写入；其他节点只读
    - topic: 仅 select_vocab_node 写入；其他节点只读
    - profile_strategy: 仅 profile_summarizer_node 写入；其他节点只读
    - profile_weakness: 仅 profile_summarizer_node 写入；其他节点只读
    - last_english_sentence: 仅 generate_sentence_node 写入；其他节点只读
    - _exposure_recorded: 仅 generate_sentence_node 标记与 persist_turn_node 读取；其他节点只读
    - non_ket_annotations: 仅 generate_sentence_node 写入；其他节点只读
    """
    messages: list[AnyMessage]
    intent: KetIntent | None
    ...
```

- [ ] **Step 3: 校验 `mypy src` 确认无类型违规**

Run: `mypy src`  
Expected: `Success: no issues found in 37 source files`

- [ ] **Step 4: 提交代码**

```bash
git add src/flow/ket_partner/state.py
git commit -m "docs(ket_partner): add single-writer docstrings and narrow KetIntent type"
```

---

### Task 3: 测试 Mock 断言补全与校验

**Files:**
- Modify: `tests/ket_partner/test_input_classifier.py`
- Modify: `tests/ket_partner/test_profile_summarizer.py`
- Modify: `tests/ket_partner/test_sentence_generator.py`
- Modify: `tests/ket_partner/test_sentence_naturalness.py`
- Modify: `tests/ket_partner/test_translation_evaluator.py`
- Modify: `tests/ket_partner/test_word_meaning_lookup.py`

- [ ] **Step 1: 给 `test_input_classifier.py` 补充 Mock `.assert_awaited_once()`**

In `tests/ket_partner/test_input_classifier.py`:
```python
    res = await classify_input_node(state, llm)
    assert res["intent"] == "translate"
    bound.ainvoke.assert_awaited_once()
```

- [ ] **Step 2: 给 `test_profile_summarizer.py` 补充 Mock `.assert_awaited_once()`**

In `tests/ket_partner/test_profile_summarizer.py`:
```python
    res = await profile_summarizer_node(state, llm)
    assert res["profile_strategy"] is not None
    bound.ainvoke.assert_awaited_once()
```

- [ ] **Step 3: 给 `test_sentence_generator.py` 补充 Mock `.assert_awaited_once()`**

In `tests/ket_partner/test_sentence_generator.py`:
```python
    res = await generate_sentence_node(state, llm)
    assert res["last_english_sentence"] is not None
    bound.ainvoke.assert_awaited()
```

- [ ] **Step 4: 给 `test_sentence_naturalness.py` 补充 Mock `.assert_awaited_once()`**

In `tests/ket_partner/test_sentence_naturalness.py`:
```python
    res = await evaluate_sentence_naturalness(sentence, llm)
    assert res is True
    bound.ainvoke.assert_awaited_once()
```

- [ ] **Step 5: 给 `test_translation_evaluator.py` 补充 Mock `.assert_awaited_once()`**

In `tests/ket_partner/test_translation_evaluator.py`:
```python
    res = await evaluate_translation_node(state, llm)
    assert res["overall_correct"] is True
    bound.ainvoke.assert_awaited()
```

- [ ] **Step 6: 给 `test_word_meaning_lookup.py` 补充 Mock `.assert_awaited_once()`**

In `tests/ket_partner/test_word_meaning_lookup.py`:
```python
    res = await lookup_target_meaning_node(state, llm)
    assert res["asked_word_meaning"] is not None
    bound.ainvoke.assert_awaited_once()
```

- [ ] **Step 7: 运行 Pytest 测试并确认通过**

Run: `pytest tests/ket_partner/`  
Expected: All tests pass successfully.

- [ ] **Step 8: 提交代码**

```bash
git add tests/ket_partner/*.py
git commit -m "test(ket_partner): reinforce mock await_count and assert_awaited assertions"
```

---

### Task 4: 第二阶段 Quality Gate 门禁验证

**Files:**
- Audit: All codebase

- [ ] **Step 1: 运行 Ruff 静态检查**

Run: `ruff check .`  
Expected: All checks passed!

- [ ] **Step 2: 运行 Mypy 静态类型检查**

Run: `mypy src`  
Expected: `Success: no issues found in 37 source files`

- [ ] **Step 3: 运行 Pytest 测试**

Run: `pytest`  
Expected: All tests pass.

- [ ] **Step 4: 提交记录**

```bash
git commit --allow-empty -m "ci: complete phase 2 code compliance quality gate check"
```
