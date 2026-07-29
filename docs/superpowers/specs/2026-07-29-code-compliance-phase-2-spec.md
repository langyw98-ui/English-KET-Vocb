# 代码规范合规改造 —— 第二阶段：`ket_partner` 核心模块 10 维规范合规治理详细规范

**文档日期**：2026-07-29  
**关联总方案**：[2026-07-29-code-compliance-refactoring-design.md](./2026-07-29-code-compliance-refactoring-design.md)  
**目标**：对 `src/flow/ket_partner/` 目录下的 20 个源码模块与 `tests/ket_partner/` 目录下的 20 个测试模块实施全面 10 维规范审计与改造。

---

## 一、异常处理纪律规范改造

[CLAUDE.md](../../CLAUDE.md) 第一条明确规定：
- 禁止使用 `except Exception as e: # noqa: BLE001` 或裸 `except`。
- 跨边界调用必须捕获具体异常元组（如 `(TimeoutError, OSError, ValidationError, KeyError, RuntimeError)`）。
- 所有兜底分支必须包含 `logger.warning(..., exc_info=True)`。

### 1.1 `src/flow/ket_partner/agent.py`
* **问题定位**：L440 `except Exception as e: # noqa: BLE001` 兜底 LLM 调用失败。
* **修改方案**：
  - 移除 `# noqa: BLE001` 压制注释。
  - 将捕获类型精确替换为具体异常元组：`except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保包含 `logger.warning(f"Agent LLM 交互异常: {e}", exc_info=True)`。

### 1.2 `src/flow/ket_partner/input_classifier.py`
* **问题定位**：L33 `except Exception as e: # noqa: BLE001`。
* **修改方案**：
  - 移除 `# noqa: BLE001`。
  - 精确捕获 `except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保记录 `logger.warning(f"输入分类节点异常: {e}", exc_info=True)`。

### 1.3 `src/flow/ket_partner/profile_summarizer.py`
* **问题定位**：L40 `except Exception as e: # noqa: BLE001`。
* **修改方案**：
  - 移除 `# noqa: BLE001`。
  - 精确捕获 `except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保记录 `logger.warning(f"用户画像总结节点异常: {e}", exc_info=True)`。

### 1.4 `src/flow/ket_partner/sentence_generator.py`
* **问题定位**：L126 `except Exception as e: # noqa: BLE001`。
* **修改方案**：
  - 移除 `# noqa: BLE001`。
  - 精确捕获 `except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保记录 `logger.warning(f"例句生成节点异常: {e}", exc_info=True)`。

### 1.5 `src/flow/ket_partner/sentence_naturalness.py`
* **问题定位**：L51 `except Exception as e: # noqa: BLE001`。
* **修改方案**：
  - 移除 `# noqa: BLE001`。
  - 精确捕获 `except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保记录 `logger.warning(f"自然度校验节点异常: {e}", exc_info=True)`。

### 1.6 `src/flow/ket_partner/translation_evaluator.py`
* **问题定位**：L129 `except Exception as e: # noqa: BLE001`。
* **修改方案**：
  - 移除 `# noqa: BLE001`。
  - 精确捕获 `except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保记录 `logger.warning(f"翻译评估节点异常: {e}", exc_info=True)`。

### 1.7 `src/flow/ket_partner/word_meaning_lookup.py`
* **问题定位**：L72, L113, L138 处的 3 处 `except Exception as e: # noqa: BLE001`。
* **修改方案**：
  - 全部移除 `# noqa: BLE001`。
  - 精确捕获 `except (TimeoutError, RuntimeError, ValueError) as e:`。
  - 确保记录 `logger.warning(..., exc_info=True)`。

---

## 二、状态与 Single-Writer Docstring 规范改造

[CLAUDE.md](../../CLAUDE.md) 第三条要求：
- 共享可变状态的字段必须在所属类型的 docstring 里逐字段声明写入者：`- field_name: 仅 <写入者> 在 <条件> 时写；<其他位置> 只读`。
- 限制有限取值集合字段必须使用 `Literal` 或 `Enum`。

### 2.1 `src/flow/ket_partner/state.py` 改造方案
1. 将开放 `intent: str | None` 类型收窄为：
   ```python
   KetIntent = Literal["translate", "asks_meaning", "idk", "off_topic", "non_compliant"]
   ```
   字段标注为 `intent: KetIntent | None`。
2. 为 `BTPKetState` 的 18 个属性补充 Single-Writer docstring 注释：
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
   ```

---

## 三、测试规范与 Mock 断言强化

[CLAUDE.md](../../CLAUDE.md) 第六条规定：
- 凡是用 `unittest.mock.patch` 或 `MagicMock` / `AsyncMock` 替换目标的单元测试，必须显式断言调用次数（`call_count` / `await_count` 或 `assert_called*` / `assert_awaited*`）。
- 确保测试符合 Hermetic 隔离要求。

### 3.1 改造测试文件清单（`tests/ket_partner/`）
对以下测试文件补充 `bound.ainvoke.assert_awaited()` / `await_count` 断言：
1. `tests/ket_partner/test_input_classifier.py`
2. `tests/ket_partner/test_profile_summarizer.py`
3. `tests/ket_partner/test_sentence_generator.py`
4. `tests/ket_partner/test_sentence_naturalness.py`
5. `tests/ket_partner/test_translation_evaluator.py`
6. `tests/ket_partner/test_word_meaning_lookup.py`

*示例修改*：
```python
# 修改前
res = await classify_input_node(...)
assert res["intent"] == "translate"

# 修改后
res = await classify_input_node(...)
assert res["intent"] == "translate"
bound.ainvoke.assert_awaited_once()  # 确保 Mock 真实被调用过
```

---

## 四、第二阶段验证命令 (Quality Gate)

修改完成后，必须依次执行：

```bash
# 1. 语法与代码格式静态校验
ruff check .

# 2. 静态类型校验（0 报错）
mypy src

# 3. 全量测试回归与 Mock 断言校验
pytest
```
