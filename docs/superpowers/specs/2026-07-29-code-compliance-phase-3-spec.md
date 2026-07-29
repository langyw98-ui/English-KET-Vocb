# 代码规范合规改造 —— 第三阶段：`english_training_partner` 拆分重构详细规范

**文档日期**：2026-07-29（修订版，替换原 nodes/ 文件夹方案）
**关联总方案**：[2026-07-29-code-compliance-refactoring-design.md](./2026-07-29-code-compliance-refactoring-design.md)
**关联规范更新**：[CLAUDE.md](../../CLAUDE.md) 第十条.1（LangGraph 三层映射 + 两类反模式）
**目标**：将 1663 行的单文件 `src/flow/english_training_partner.py` 拆解为按"业务编排 / 决策路由 / 外部调用"三层正交分层的子包 `src/flow/english_training_partner/`，对齐 CLAUDE.md 第十条.1。

---

## 一、改造目标与原方案修正

### 1.1 原方案问题
原 spec 提出 `nodes/` 子包按节点类别拆 6 个文件（init_node.py / negotiation_node.py / teaching_nodes.py / dialogue_node.py / exit_node.py / input_classifier_node.py）。该方案违反 CLAUDE.md 第十条.1：

- **按节点类别拆文件夹**混淆了"图层"与"领域层"——节点是图抽象的顶点，不归任何单一领域
- **`teaching_nodes.py` 把 5 个不同职责函数塞一起**（vocab/sentence/grammar/compliance/explain）只是把 god class 换成 god file
- **`negotiation_handler`（333 行）和 `input_classifier`（215 行）原样搬移**没解决 god function 问题

### 1.2 修正后的三层架构（对齐 CLAUDE.md 第十条.1）

| 正交职责 | 文件 | 内容 |
|---------|------|------|
| 业务编排 | `nodes.py` | 14 个 LangGraph node 薄壳，仅做 state IO + 调领域模块 |
| 决策路由 | `graph.py` | `BTPAutonomous.compile` + 3 个 `route_*` 函数 + 工厂 |
| 外部调用 | `scene.py` / `classification.py` / `negotiation.py` / `teaching.py` / `explanation.py` / `compliance.py` / `dialogue.py` / `exit_reason.py` | LLM/IO 调用与领域计算 |

### 1.3 兼容性
全仓库扫描确认 `english_training_partner` 当前**无外部消费者**（`src/api` 不引用、无测试文件），故不保留 Facade，直接删除原 `.py`。

---

## 二、包结构

```
src/flow/english_training_partner/
├── __init__.py               # 包标记 + 导出 autonomous / BTPState / BTPAutonomous
├── state.py                  # BTPState TypedDict + Single-Writer docstring + Literal 收窄
├── schemas.py                # Pydantic 契约（LLM 结构化输出）+ 内部 dataclass 值对象
├── constants.py              # _FALLBACK_SCENES / DEFAULT_TOOL_REPLIES / 阈值常量
├── graph.py                  # BTPAutonomous + NodeContext + 3 route_* + autonomous 工厂
├── nodes.py                  # 14 个 node 薄壳（partial 绑定 NodeContext）
├── scene.py                  # 场景加载与生成业务
├── classification.py         # 意图分类 + 输入分类业务（含 god function 拆解）
├── negotiation.py            # 协商业务（含 god function 拆解）
├── teaching.py               # 词汇/句子/语法教学业务
├── explanation.py            # 释义业务
├── compliance.py             # 合规重定向业务
├── dialogue.py               # 工具执行 + 工具回复 + 对话响应业务
└── exit_reason.py            # 退出处理业务
```

---

## 三、NodeContext 模式（解决 node 访问依赖）

node 函数签名必须为 `(state) -> state`（LangGraph 契约），但 node 内部要调领域模块，领域模块需要 model/tools/scenes 等依赖。采用 `NodeContext` dataclass + `functools.partial` 绑定：

```python
# graph.py
@dataclass(frozen=True, slots=True)
class NodeContext:
    model: BaseChatModel
    model_smart: BaseChatModel
    model_with_tools: Any
    tools: dict[str, Any]
    scenes: list[dict]
    system_prompt: SystemMessage
    info: dict

class BTPAutonomous:
    async def compile(self, builder, checkpointer):
        ctx = NodeContext(...)
        builder.add_node("init_state", partial(nodes.init_state, ctx=ctx))
        builder.add_node("intent_classifier", partial(nodes.intent_classifier, ctx=ctx))
        # ... 其余 12 个
```

```python
# nodes.py
async def init_state(state: BTPState, *, ctx: NodeContext) -> BTPState:
    """薄壳：仅做 state IO 编排，业务在 scene.py。"""
    state = scene.initialize_state(state, scenes=ctx.scenes)
    return state
```

---

## 四、详细文件规范

### 4.1 `state.py` —— 状态定义

**职责**：定义 `BTPState` TypedDict，所有字段收窄为 `Literal` 或显式可选类型，逐字段标注 Single-Writer。

**类型定义**：
```python
class BTPState(TypedDict):
    messages: list[AnyMessage]
    intent: Literal["tool", "flow_resp"] | None
    next_action_node: Literal[
        "negotiation_handler", "input_classifier", "exit_in_reason", None
    ]
    phase: Literal["negotiation", "dialogue"] | None
    mode: Literal["scene", "free_chat"] | None
    scene_category: str | None
    scene_name: str | None
    ai_role: str | None
    user_role: str | None
    proposed_scene: dict | None
    input_type: Literal[
        "non_compliant", "vocab_gap", "sentence_gap",
        "grammar_error", "all_chinese", "ask_explain", "clean"
    ] | None
    teach_words: str | None
    correct_sentence: str | None
    error_explain: str | None
    error_counts: dict[str, int] | None
    awaiting_scene_selection: bool | None
    valid_dialogues: list[AnyMessage]
    tool_result: str | None
```

**Single-Writer docstring 逐字段声明**（按 CLAUDE.md 第三条.3）：
- `messages`: 仅各 node 在处理用户输入时 append；其他位置只读
- `intent`: 仅 `nodes.intent_classifier` 写；其他位置只读
- `next_action_node`: 仅协商/教学/分类相关 node 写；route_* 只读
- `phase`: 仅 `negotiation_handler` /首次 init 写；其他只读
- `mode`: 仅 `negotiation_handler`（确认场景或 free_chat 时）写
- `proposed_scene`: 仅 `negotiation_handler` 写
- `input_type` / `teach_words` / `correct_sentence` / `error_explain`: 仅 `nodes.input_classifier` 写
- `error_counts`: 仅 `nodes.input_classifier` 写
- `awaiting_scene_selection`: 仅 `negotiation_handler` 写
- `valid_dialogues`: 仅 `negotiation_handler` / `dialogue_respond` 写
- `tool_result`: 仅 `tool_executor` / `exit_in_reason` 写

---

### 4.2 `schemas.py` —— 数据契约

**职责**：隔离 LLM 结构化输出 Pydantic 模型 + 跨函数边界的内部 dataclass 值对象。

**LLM 结构化输出模型**（沿用原文件 line 61-96）：
- `AnalysisOutput(BaseModel)`: teach_words / correct_sentence / error_explain —— 教学分析
- `ErrorKeyOutput(BaseModel)`: error_key —— 标准化错误标识
- `CustomSceneOutput(BaseModel)`: name / category / ai_role / user_role / reference_opening —— 自定义场景
- `NegotiationIntent(BaseModel)`: intent (Literal 8 值) + confidence —— 协商分类

**内部值对象**（dataclass，CLAUDE.md 第二条.11）：
- `InputClassification` (frozen dataclass): input_type / teach_words / correct_sentence / error_explain / error_key / error_counts（updated dict）
- `NegotiationDecision` (frozen dataclass): ai_messages (list[AIMessage]) / state_updates (dict) / next_action_node
- `ToolExecutionResult` (frozen dataclass): messages (list[AnyMessage]) / tool_result (str | None)
- `SceneProposal` (frozen dataclass): scene (dict) / proposal_msg (str)

---

### 4.3 `constants.py` —— 静态配置

**职责**：模块级命名常量，禁止业务函数内联。

**常量清单**：
- `_FALLBACK_SCENES: list[dict]` —— 兜底场景列表（原 line 123-145）
- `DEFAULT_TOOL_REPLIES: dict[str, str]` —— 工具名 → 默认回复（原 line 566-584，从 set_tool_call_hint 内联提取）
- `DEFAULT_CHAT_REPLY: str = "好的，我明白了。"` —— 通用默认回复
- `DEFAULT_CHAT_REPLY_AFTER_TOOL: str = "操作已完成。"` —— 工具后默认回复
- `NEGOTIATION_CONFIDENCE_THRESHOLD: float = 0.6` —— 协商低置信度兜底（原 line 872）
- `ERROR_TOLERANCE_THRESHOLD: int = 3` —— 同类错误容忍上限（原 line 1236）
- `DIALOGUE_CONTEXT_WINDOW: int = 5` —— valid_dialogues 上下文窗口（多处 -5）
- `SCENE_DESC_MAX_LEN: int = 30` —— 场景描述字数上限
- `VALID_INPUT_TYPES: frozenset[str]` —— input_type 合法集合（用于运行时校验）

---

### 4.4 `graph.py` —— 图编排与路由

**职责**：`BTPAutonomous` 类只保留 `__init__` + `compile`；3 个路由条件函数与 1 个工厂函数为模块顶层。

**`BTPAutonomous` 类**（瘦身自原 22 方法）：
- `__init__(self, model, model_smart, tools, info, system_prompt)`: 仅赋值实例属性（model_with_tools/original_model/original_model_smart/tools/system_prompt/info/is_init_state/scenes/_last_proposed_name）
- `async compile(self, builder, checkpointer) -> CompiledStateGraph`: 构建 NodeContext + `add_node` 注册 14 个节点（partial 绑定 ctx）+ `add_edge` / `add_conditional_edges` 连接图拓扑

**路由条件函数**（顶层）：
- `route_intent(state) -> Literal["tool", "flow_resp"]`: 根据 `state["intent"]` 路由（原 line 416-417）
- `async route_by_phase(state, ctx) -> Literal["negotiation_handler", "input_classifier", "exit_in_reason"]`: 三路分支——next_action_node 显式指定时直接走；为 None 时调 LLM 判 reject/continue（原 line 611-637）
- `route_input_type(state) -> str`: 6 路分支——根据 input_type 分发到 compliance_redirect / teach_vocab / teach_sentence / teach_grammar / explain_meaning / dialogue_respond（原 line 1579-1591）

**工厂函数**：
- `async autonomous(info: dict) -> CompiledStateGraph`: 创建 LLM 实例 + tools + system_prompt，构造 BTPAutonomous，返回编译后的图（原 line 1597-1610）

---

### 4.5 `nodes.py` —— 14 个 LangGraph node 薄壳

**职责**：每个 node 仅做 state 读 + 调领域模块 + state 写，单函数 ≤ 30 行。

**节点清单**：

| 节点函数 | 签名 | 调用的领域函数 | 来源方法 |
|---------|------|--------------|---------|
| `init_state` | `async (state, *, ctx) -> state` | `scene.initialize_state(state, scenes)` | 原线 248-276 |
| `intent_classifier` | `async (state, *, ctx) -> state` | `classification.classify_intent(ctx.model_with_tools, user_msg)` | 原线 359-412 |
| `tool_executor` | `async (state, *, ctx) -> state` | `dialogue.execute_tool(ctx.model_with_tools, ctx.tools, messages)` | 原线 421-476 |
| `set_tool_call_hint` | `async (state, *, ctx) -> state` | `dialogue.build_tool_reply(ctx.model, messages)` | 原线 478-600 |
| `flow_resp` | `async (state, *, ctx) -> state` | （no-op，仅 log） | 原线 604-607 |
| `exit_in_reason` | `async (state, *, ctx) -> state` | `exit_reason.handle_exit(ctx.model_with_tools, ctx.tools, messages)` | 原线 639-701 |
| `negotiation_handler` | `async (state, *, ctx) -> state` | `negotiation.negotiate(ctx.model, ctx.scenes, state, user_msg, ctx._last_proposed_name)` | 原线 705-1034 |
| `input_classifier` | `async (state, *, ctx) -> state` | `classification.classify_input(ctx.model, ctx.model_smart, user_msg, messages, state["error_counts"])` | 原线 1038-1249 |
| `teach_vocab` | `async (state, *, ctx) -> state` | `teaching.teach_vocab_message(ctx.model_smart, user_msg, teach_words, correct_sentence, valid_dialogues)` | 原线 1253-1300 |
| `teach_sentence` | `async (state, *, ctx) -> state` | `teaching.teach_sentence_message(...)` | 原线 1304-1353 |
| `teach_grammar` | `async (state, *, ctx) -> state` | `teaching.teach_grammar_message(...)` | 原线 1357-1402 |
| `compliance_redirect` | `async (state, *, ctx) -> state` | `compliance.redirect_message(ctx.model_smart, phase, ai_role, valid_dialogues, messages)` | 原线 1406-1464 |
| `dialogue_respond` | `async (state, *, ctx) -> state` | `dialogue.dialogue_reply(ctx.model, mode, ai_role, valid_dialogues, user_msg)` | 原线 1468-1522 |
| `explain_meaning` | `async (state, *, ctx) -> state` | `explanation.explain_message(ctx.model, messages, user_msg)` | 原线 1526-1575 |

**节点模式示例**：
```python
async def teach_vocab(state: BTPState, *, ctx: NodeContext) -> BTPState:
    logger.debug("【NODE】teach_vocab")
    current_messages = list(state["messages"])
    user_message = current_messages[-1].content if current_messages else ""
    teaching = await teaching_mod.teach_vocab_message(
        ctx.model_smart,
        user_message=user_message,
        teach_words=state.get("teach_words") or "",
        correct_sentence=state.get("correct_sentence") or "",
        valid_dialogues=list(state.get("valid_dialogues") or []),
    )
    current_messages.append(AIMessage(content=teaching))
    state["messages"] = current_messages
    state["next_action_node"] = "input_classifier"
    return state
```

---

### 4.6 `scene.py` —— 场景业务

**职责**：场景库加载与初始化、场景描述生成、自定义场景生成、场景列表文案。

**函数清单**：
- `initialize_state(state: BTPState, *, scenes: list[dict]) -> BTPState`: 首次调用时初始化所有 state 字段为 None/默认值；后续调用为 no-op。**注意**：原代码此分支用 `self.is_init_state` 实例标志判断，重构后改为 `state.get("phase") is None` 检测（避免可变实例状态）
- `load_scenes() -> list[dict]`: 从 `data/scenes.json` 异步加载（用 `aiofiles` 或 `asyncio.to_thread`），失败兜底 `_FALLBACK_SCENES`。**原 line 278-291 的同步 `open` 违反 CLAUDE.md 第七条.2，必须异步化**
- `async generate_scene_desc(model, scene: dict) -> str`: LLM 生成场景描述句（原 line 293-323）
- `async generate_custom_scene(model, user_desc: str) -> dict | None`: LLM 结构化生成自定义场景（原 line 325-355）
- `format_scenes_by_category(scenes: list[dict]) -> str`: 按类别格式化场景列表文案（原 line 908-916 内联逻辑提取）
- `format_scenes_numbered(scenes: list[dict]) -> str`: 编号格式化场景列表（原 line 734-737 / 943-946 重复逻辑提取）

---

### 4.7 `classification.py` —— 分类业务（含 god function 拆解）

**职责**：意图分类（tool vs flow_resp）+ 输入多维度分类（合规 / ask_explain / input_type / 教学分析 / 错误键 / 容忍度）。

**入口函数**：
- `async classify_intent(model_with_tools, user_message: str) -> Literal["tool", "flow_resp"]`: 单步意图分类，异常兜底 flow_resp（原 line 359-412 业务部分）
- `async classify_input(model, model_smart, user_message: str, current_messages: list, error_counts: dict | None) -> InputClassification`: 输入分类主入口，编排 6 个子分类步骤，返回 `InputClassification` 值对象（原 line 1038-1249 拆解）

**子分类 helper**（仅 classify_input 内部调用，模块私有 `_` 前缀）：
- `async _check_compliance(model, user_message: str) -> Literal["safe", "unsafe"]`: 合规检测（原 line 1044-1065）
- `async _check_ask_explain(model, user_message: str, prev_ai_msg: str) -> bool`: 询问释义检测（原 line 1080-1100）
- `async _classify_input_type(model, user_message: str, has_chinese: bool) -> str`: 输入类型分类（原 line 1108-1148）
- `async _analyze_with_llm(model_smart, input_type: str, user_message: str) -> AnalysisOutput | None`: 教学分析（原 line 1156-1185）
- `async _standardize_error_key(model, user_message, input_type, teach_words, correct_sentence, error_explain) -> str | None`: 错误键标准化（原 line 1190-1223）
- `_apply_error_tolerance(error_counts: dict | None, error_key: str | None, input_type: str) -> ToleranceDecision`: 同类错误容忍度决策（原 line 1226-1243）。`ToleranceDecision` dataclass: `final_input_type: str`（可能被 override 为 "clean"）+ `updated_error_counts: dict[str, int]`

**正则常量**：
- `_CHINESE_REGEX = re.compile(r"[一-鿿]")` —— 中文检测（原 line 1072）

---

### 4.8 `negotiation.py` —— 协商业务（含 god function 拆解）

**职责**：场景协商全流程——首次提议 / 后续分类 / 列表选择 / reroll / free_chat / 自定义场景 / 开场白生成。

**入口函数**：
- `async negotiate(model, scenes, state, user_message: str, last_proposed_name: str | None) -> NegotiationDecision`: 协商主入口，编排所有子分支，返回 `NegotiationDecision` 值对象（messages + state_updates + next_action_node）

**子分支 helper**（按 classification 分支拆解）：
- `async _classify_negotiation_intent(model, user_message: str) -> NegotiationIntent`: 结构化分类（原 line 838-870）
- `async _propose_default_scene(model, scenes) -> SceneProposal`: 首次提议默认场景（原 line 710-724）
- `async _handle_scene_selection(model, scenes, user_message: str, awaiting: bool, last_proposed_name) -> SelectionResult`: 处理 awaiting_scene_selection 路径（原 line 732-835）
- `async _handle_reroll(model, scenes, last_proposed_name) -> SceneProposal`: 换场景（原 line 889-906）
- `_build_list_scenes_message(scenes) -> str`: 列表场景文案（原 line 908-920）
- `_build_free_chat_message() -> str`: free_chat 文案（原 line 930-934，常量化）
- `async _generate_opening_line(model, chosen_scene: dict) -> str`: LLM 生成开场白变体（原 line 1004-1017）
- `_build_scene_confirm_msg(chosen_scene: dict, opening_line: str) -> str`: 场景确认话术（原 line 1019-1027，ai_role == user_role 与否两种模板）
- `_build_low_confidence_message() -> str`: 低置信度澄清话术（原 line 873，常量化）
- `_build_non_compliant_message() -> str`: 协商期不合规话术（原 line 883，常量化）
- `_build_custom_request_message() -> str`: 自定义场景请求话术（原 line 983-984，常量化）

**值对象**（schemas.py 中定义）：
- `SceneProposal` (frozen dataclass): scene / proposal_msg
- `SelectionResult` (frozen dataclass): chosen_scene / confirm_msg / state_updates / next_action_node / matched (bool)

---

### 4.9 `teaching.py` —— 教学业务

**职责**：词汇/句子/语法三种教学话术生成。

**函数清单**：
- `async teach_vocab_message(model_smart, user_message, teach_words, correct_sentence, valid_dialogues) -> str`: 词汇教学话术（原 line 1253-1300 业务部分）
- `async teach_sentence_message(model_smart, user_message, teach_words, correct_sentence, valid_dialogues, input_type) -> str`: 句子教学话术（原 line 1304-1353）。`input_type == "all_chinese"` 与否决定 desc
- `async teach_grammar_message(model_smart, user_message, correct_sentence, error_explain, valid_dialogues) -> str`: 语法教学话术（原 line 1357-1402）

**共享 helper**：
- `_format_dialogue_context(valid_dialogues: list) -> str`: 格式化最近 N 条对话为上下文字符串（原 line 1262-1266 / 1313-1317 / 1366-1370 三处重复逻辑提取）

**Prompt 模板常量**（按 CLAUDE.md 第四条.3）：
- `_TEACH_VOCAB_PROMPT`、`_TEACH_SENTENCE_PROMPT_TEMPLATE`、`_TEACH_GRAMMAR_PROMPT`

---

### 4.10 `explanation.py` —— 释义业务

**职责**：用户询问 AI 上一句英文含义时，用中文解释。

**函数清单**：
- `async explain_message(model, current_messages: list, user_message: str) -> str`: 找上一条 AIMessage + LLM 生成中文解释（原 line 1526-1575 业务部分）。找不到上一条时返回引导文案（原 line 1541-1542，常量化）

**Prompt 模板常量**：
- `_EXPLAIN_MEANING_PROMPT_TEMPLATE`

---

### 4.11 `compliance.py` —— 合规重定向业务

**职责**：根据 phase（dialogue / negotiation）生成合规重定向话术。

**函数清单**：
- `async redirect_message(model_smart, phase: Literal["dialogue", "negotiation"], ai_role: str, valid_dialogues: list, current_messages: list) -> str`: 双分支话术生成（原 line 1406-1464 业务部分）

**Prompt 模板常量**：
- `_COMPLIANCE_DIALOGUE_PROMPT_TEMPLATE`、`_COMPLIANCE_NEGOTIATION_PROMPT`

---

### 4.12 `dialogue.py` —— 对话业务

**职责**：工具执行 + 工具回复生成 + 常规对话响应。

**函数清单**：
- `async execute_tool(model_with_tools, tools: dict, current_messages: list) -> ToolExecutionResult`: 调用模型 + 执行工具调用 + 构造 ToolMessage（原 line 421-476 业务部分）
- `async build_tool_reply(model, current_messages: list) -> list[AnyMessage]`: 工具回复生成——优先用工具预设 reply_content，否则调 LLM 生成，空回复时按工具名查表兜底（原 line 478-600 业务部分）
- `async dialogue_reply(model, mode: Literal["scene", "free_chat"], ai_role: str, valid_dialogues: list, user_message: str) -> str`: 角色扮演或自由聊天的对话响应（原 line 1468-1522 业务部分）

**Prompt 模板常量**：
- `_TOOL_REPLY_CHAT_PROMPT`、`_MEMORY_PROMPT_TEMPLATE`、`_DIALOGUE_SCENE_PROMPT_TEMPLATE`、`_DIALOGUE_FREE_CHAT_PROMPT`

**辅助函数**：
- `_extract_preset_reply(current_messages: list) -> str | None`: 从 ToolMessage 解析 expect_reply/reply_content（原 line 488-509）
- `_find_last_tool_name(current_messages: list) -> str | None`: 提取最后调用的工具名（原 line 557-563）

---

### 4.13 `exit_reason.py` —— 退出处理业务

**职责**：触发 exit 工具调用 + 追加退出提示。

**函数清单**：
- `async handle_exit(model_with_tools, tools: dict, current_messages: list) -> list[AnyMessage]`: 调用模型让其触发 exit 工具，执行工具，追加"已退出英语场景练习模式。"消息（原 line 639-701 业务部分）

**常量**：
- `_EXIT_MESSAGE: str = "已退出英语场景练习模式。"`

---

## 五、God function 拆解策略

### 5.1 `negotiation_handler`（333 行 → 主入口 ≤ 60 行 + 11 helper）

**拆解维度**：按协商状态机的分支切割。原函数包含 7 个 classification 分支 + awaiting_scene_selection 子路径 + 首次提议路径 + 场景确认收尾。

**新结构**：
```
negotiate()  # 主入口：if proposed_scene is None → _propose_default_scene
             #         elif awaiting → _handle_scene_selection
             #         else → _classify_negotiation_intent → switch by 8 classifications
             # 收尾：所有确认场景的分支共用 _generate_opening_line + _build_scene_confirm_msg
```

每个 helper ≤ 80 行；主入口 ≤ 60 行（switch 路由 + 共用收尾）。

### 5.2 `input_classifier`（215 行 → 主入口 ≤ 50 行 + 6 helper）

**拆解维度**：按 5 步分类流程切割（compliance → ask_explain → input_type → analysis → error_key → tolerance）。

**新结构**：
```
classify_input()  # 主入口：6 步顺序编排，逐步填充 InputClassification
                  # 短路：compliance unsafe → InputClassification.input_type = "non_compliant"，跳过后续步骤
                  #      ask_explain True → input_type = "ask_explain"，跳过后续步骤
                  # 步骤 6 _apply_error_tolerance 可能将 input_type override 为 "clean"（并触发教学字段清空）
                  # 返回 InputClassification 值对象，由 node 写回 state
```

### 5.3 `set_tool_call_hint`（126 行 → 主入口 ≤ 40 行 + 2 helper）

**拆解维度**：预设回复路径 / LLM 生成路径 / 空回复兜底路径。

**新结构**：见 4.12 `build_tool_reply` —— 主入口先 `_extract_preset_reply`，命中则直接构造 AIMessage 返回；否则调 LLM；空回复时 `_find_last_tool_name` 查表兜底。

---

## 六、类型契约新增

### 6.1 State 字段 Literal 收窄
见 4.1，所有有限取值字段（intent / phase / mode / input_type / next_action_node）从 `str` 收窄为 `Literal`。

### 6.2 值对象（dataclass）
见 4.2，跨函数边界返回值全部使用 frozen dataclass，禁止裸多元组。

### 6.3 LLM 调用统一结构化
所有 LLM 调用必须用 `with_structured_output(Schema, method="function_calling")` 或视为"纯文本回复"（如对话响应）。原代码中部分裸 `.content` 文本调用（compliance/ask_explain/explain 等）维持文本调用但增加明确校验。

---

## 七、异常处理策略（对齐 CLAUDE.md 第一条.5）

### 7.1 异常元组收紧
原代码所有 try 块均为 `except Exception`，违反 CLAUDE.md 第一条.1/1.5。重构时按调用边界区分：

| 调用类型 | 允许捕获的异常 |
|---------|--------------|
| LLM 调用（`model.ainvoke` / `with_structured_output`） | `(asyncio.TimeoutError, RuntimeError)` + langchain 显式异常 |
| 文件 IO（`load_scenes`） | `(OSError, json.JSONDecodeError)` |
| 工具执行（`tool.ainvoke`） | 工具自定义异常 + `(asyncio.TimeoutError, RuntimeError)` |
| JSON 解析（`_extract_preset_reply`） | `(json.JSONDecodeError, TypeError, AttributeError)` —— 仅在 Parser 适配层内允许 |

**禁止捕获**：`ValueError` / `KeyError` / `IndexError` / `AttributeError` 在业务层（让其暴露代码 bug）。例外：Parser 适配层（`_extract_preset_reply` / `_parse_*`）允许捕获并重新封装为业务定义类型。

### 7.2 Fallback 必须有日志
所有 fallback 分支必须 `logger.warning(..., exc_info=True)`，不允许静默兜底（对齐 CLAUDE.md 第一条.3）。

### 7.3 异步文件 IO
`scene.load_scenes` 必须用 `aiofiles` 或 `asyncio.to_thread`，禁止 async 函数内同步 `open`（对齐 CLAUDE.md 第七条.2）。

---

## 八、测试策略

### 8.1 测试目录
新建 `tests/english_training_partner/`，按被测单元拆分（CLAUDE.md 第六条.3）：

```
tests/english_training_partner/
├── __init__.py
├── conftest.py                    # 共享 fixtures（NodeContext / mock model 等）
├── test_state.py                  # BTPState Literal 校验
├── test_schemas.py                # Pydantic 模型校验
├── test_graph.py                  # compile 注册的节点 + 边
├── test_nodes.py                  # 14 个 node 薄壳（mock 领域模块）
├── test_scene.py                  # scene 业务
├── test_classification.py         # classification 业务（含 god function 拆解后的所有 helper）
├── test_negotiation.py            # negotiation 业务（含 8 classification 分支 + awaiting 路径）
├── test_teaching.py               # 3 个教学话术
├── test_explanation.py            # 释义业务
├── test_compliance.py             # 合规重定向（dialogue + negotiation 双分支）
├── test_dialogue.py               # 工具执行 + 工具回复 + 对话响应
└── test_exit_reason.py            # 退出处理
```

### 8.2 Mock 断言要求（CLAUDE.md 第六条.4-5）
所有 mock 测试必须断言：
- 同步方法：`assert_called*` / `call_count`
- 异步方法：`assert_awaited*` / `await_count`

禁止仅断言返回值——fallback 默认值与正常返回值形态重合时，仅断言返回值无法区分两条路径。

### 8.3 集成测试
真实 LLM 调用测试标记 `@pytest.mark.integration`，缺 API key 时 skip。

---

## 九、质量门禁（CLAUDE.md 第九条.1）

重构完成后，必须依次执行且全部清零：

```bash
ruff check .
mypy src
pytest
```

mypy 必须开启 `strict = true` 或关键检查（`disallow_untyped_defs` / `warn_return_any` / `no_implicit_optional`）。

---

## 十、迁移步骤（供 writing-plans 参考）

1. 创建 `src/flow/english_training_partner/` 包目录与空 `__init__.py`
2. 迁移 `state.py` + `schemas.py` + `constants.py`（纯类型与常量，无依赖）
3. 迁移 `graph.py` 的 `BTPAutonomous` 类骨架与 `NodeContext`
4. 按依赖顺序逐个迁移领域模块：`scene` → `explanation` → `compliance` → `teaching` → `dialogue` → `exit_reason` → `classification` → `negotiation`（依赖多的后迁）
5. 迁移 `nodes.py` 的 14 个薄壳
6. 在 `graph.py.compile` 中用 partial 绑定并注册节点
7. 迁移 `autonomous` 工厂函数
8. 删除原 `english_training_partner.py`
9. 补齐 `tests/english_training_partner/` 全部测试
10. 跑质量门禁三项清零
