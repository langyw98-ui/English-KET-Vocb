# 代码规范合规改造 —— 第三阶段：`english_training_partner.py` 模块化拆分与重构详细规范

**文档日期**：2026-07-29  
**关联总方案**：[2026-07-29-code-compliance-refactoring-design.md](./2026-07-29-code-compliance-refactoring-design.md)  
**目标**：将 1661 行的单文件 `src/flow/english_training_partner.py` 拆解为高内聚、低耦合的独立包 `src/flow/english_training_partner/`，消灭超长文件与超大类，对齐 [CLAUDE.md](../../CLAUDE.md) 第四条（代码组织）与第十条（模块分层与依赖方向）。

---

## 一、拆分目标与架构设计

### 1.1 背景与现状
当前 `src/flow/english_training_partner.py` 包含 1661 行代码，内部定义了：
- `BTPState` TypedDict 状态定义
- 5 个 Pydantic 结构化输出 Schema（`AnalysisOutput` / `ErrorKeyOutput` 等）
- 1 个包含 22 个方法的超大类 `BTPAutonomous`（涵盖路由逻辑、协商逻辑、分类逻辑、教学逻辑、对话逻辑等多种正交职责）

这严重违背了代码规范中“函数不超过 80 行、模块单一职责、数据契约与业务逻辑分离”的要求。

### 1.2 模块化子包结构

我们将 `src/flow/english_training_partner.py` 升级为 `src/flow/english_training_partner/` 包：

```
src/flow/english_training_partner/
├── __init__.py                   # 导出核心接口 (autonomous, BTPState, BTPAutonomous 等)
├── state.py                      # BTPState 状态定义（附 Single-Writer docstrings）
├── schemas.py                    # Pydantic 契约模型 (AnalysisOutput, ErrorKeyOutput 等)
├── constants.py                  # 工具定义、默认场景查表与静态配置
├── nodes/                        # 节点函数包 (每个节点模块/函数 <= 80 行)
│   ├── __init__.py
│   ├── init_node.py              # 初始化与场景选择节点
│   ├── negotiation_node.py       # 破冰协商节点
│   ├── input_classifier_node.py  # 意图/输入分类节点
│   ├── teaching_nodes.py         # 词汇/句子/语法/释义教学节点
│   ├── dialogue_node.py          # 对话与工具回调处理节点
│   └── exit_node.py              # 退出处理节点
└── graph.py                      # LangGraph 编排、路由条件函数与 BTPAutonomous 图引擎
```

同时保留原 `src/flow/english_training_partner.py` 作为 **Facade（门面）**，统一 re-export 拆分后的核心符号，确保对外部 `src/api` 及单元测试透明无缝兼容。

---

## 二、子模块职责与规范要求

### 2.1 `state.py` —— 状态定义与 Single-Writer 声明
- 包含 `BTPState(TypedDict)`，为其字段补充 Single-Writer docstrings。
- 对有限可选值的状态字段补充 `Literal` 类型约束。

### 2.2 `schemas.py` —— Pydantic 数据契约隔离
- 隔离 `AnalysisOutput`, `ErrorKeyOutput`, `CustomSceneOutput`, `NegotiationIntent`, `Task` 结构化 Schema。

### 2.3 `nodes/*.py` —— 单一职责节点
- **`init_node.py`**：处理 `init_state`、`_load_scenes`、`_generate_scene_desc`、`_generate_custom_scene`。
- **`negotiation_node.py`**：处理协商阶段对话 `negotiation_handler`。
- **`input_classifier_node.py`**：处理多维度输入分类 `input_classifier`。
- **`teaching_nodes.py`**：处理 `teach_vocab`, `teach_sentence`, `teach_grammar`, `compliance_redirect`, `explain_meaning`。
- **`dialogue_node.py`**：处理 `set_tool_call_hint`, `tool_executor`, `flow_resp`, `dialogue_respond`。
- **`exit_node.py`**：处理 `exit_in_reason`。

每个 Node 函数行数严格控制在 80 行以内；跨边界调用具备显式 `except (TimeoutError, RuntimeError, ValueError)` 异常隔离。

### 2.4 `graph.py` —— 图编排与路由
- 实现 `BTPAutonomous` 类的 `compile` 方法与 3 个路由条件方法（`route_intent`, `route_by_phase`, `route_input_type`）。
- 提供全局工厂函数 `async def autonomous(info: dict) -> CompiledStateGraph`。

---

## 三、第三阶段验证命令 (Quality Gate)

重构完成后，必须依次执行：

```bash
# 1. 代码格式与 Lint 校验
ruff check .

# 2. 静态类型校验（必须 0 报错）
mypy src

# 3. 全量测试回归
pytest
```
