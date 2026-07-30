# 代码规范合规修复设计 (2026-07-30)

## 背景

`refactor/code-compliance` 分支上完成了一轮全项目代码规范审核,按 `CLAUDE.md` 九个章节
对照共发现 24 项违规,经复核后其中 2 项为代理误判(`sentence_domain.py:207` 模块级同步
IO、`flow/common.py:30` 同步函数内同步 IO——两者均不在 async 上下文,非 §7.2 违规),
实际需修复 **22 项** = 8 P0 + 4 P1 + 10 P2。

本设计文档基于用户提供的 `docs/superpowers/raw/代码规范问题及修改意见.txt`,经 5 轮
brainstorming 一问一答确认关键决策后产出,作为后续 `writing-plans` 阶段的输入。

## 范围

### In scope

用户文档列出的 24 项违规修复(扣除 2 项误判):

- **P0 关键问题(8 项)**:类型契约不一致、LLM 异常元组混入代码 bug 类型、`app.py` 裸
  except、跨边界裸 4 元组、LLM 服务硬编码无 DI、`apply_multiword_target_patch` 修改
  入参、`generate_with_fallback` 超长函数、`agent.py` 浅层透传。
- **P1 主要问题(4 项)**:`state.py` docstring 写入者声明失真、4 处 intent 魔法字符
  串、`chat.py` 函数多职责、6 处 fallback 测试缺 mock 调用断言。
- **P2 次要问题(10 项)**:`chat_logger` 资源释放、`exporter` / `bootstrap` 异步 IO、
  `test_chat_real_llm` 未用 import、`config.py` 同步加载、`state.get("intent")` 未判
  None、`format_output_text` 分支不完备、`repos.py` 静默 no-op、`bootstrap.py` CSV
  strip 校验等。

### Out of scope

- 不重构 LangGraph 三层结构(`nodes` / `graph` / `domain` 划分保留)
- 不重写领域模块的业务逻辑
- 不动 SQL schema、LLM prompt 文案
- 不引入新依赖(已有 `aiofiles` 则用,否则 `asyncio.to_thread`)
- 不修复"撤销的 2 条误判"(`sentence_domain.py:207`、`flow/common.py:30`)

## 关键决策(brainstorming 已确认)

1. **LLM 异常元组每模块自定义**:每个领域模块(`vocab_domain` / `dialogue_domain` /
   `sentence_domain` / `agent`)定义自己的 `_LLM_RETRYABLE` 模块级常量元组,模块内
   所有 LLM 调用共享。元组内容必须按 §1.5 严格区分外部失败 vs 代码 bug:保留
   `openai.APIError` / `asyncio.TimeoutError` / `pydantic.ValidationError` 等具体外部
   失败类型,移除 `ValueError` / `AttributeError` / `TypeError`。

2. **`agent.py` 命运——方案 A:合并 `nodes.py` 入类**。`KETPartnerAgent` 类的 12 个
   节点方法从透传 wrapper 改为真实实现(直接 `await classify_intent(...)` 等,通过
   `self.llm_smart` / `self.config` 访问上下文)。删除 `nodes.py`,`graph.py` 保持
   `builder.add_node("init_state", agent.init_state)` 不变。

3. **`LlmService` Protocol + DI**:在 `flow/common.py` 定义 `LlmService` Protocol
   (`smart` / `flash` 属性)+ `DashScopeLlmService` 具体类 + 模块级
   `default_llm_service` 单例。`build_agent(llm_service: LlmService | None = None)`
   接受可选参数,默认用单例。`KETPartnerAgent.__init__(self, llm_service, config)`,
   通过 `@property` 暴露 `llm_smart` / `llm_flash`。

4. **Intent 魔法字符串——常量 + 路由 dict**:`state.py` 加常量
   `TRANSLATION` / `IDK` / `ASKS_MEANING` / `OFF_TOPIC` / `NON_COMPLIANT: KetIntent`;
   `graph.py` 两个路由函数改用 `_ROUTE_AFTER_CLASSIFY: dict[KetIntent, str]` 查表;
   `vocab_domain.apply_mastery_updates` 与 `dialogue_domain.format_output_text` 用
   常量替换字面量(复杂分支不强行查表,只换字面量为常量)。

5. **`chat_logger.py` 上下文管理器模式**:`ChatLogger` 实现 `__enter__` / `__exit__`,
   `__exit__` 兜底调 `close_session()`;调用点 `cli/ket_partner/main.py` 改
   `with ChatLogger() as logger: ...`。

6. **`app.py` shutdown 异常元组细化**:
   - line 70 (`agent.aclose`): `except (RuntimeError, asyncio.TimeoutError)`
   - line 77 (`db.close`): `except (RuntimeError, OSError, aiosqlite.Error)`

7. **撤销 2 条 P2 误判**:`sentence_domain.py:207` 模块级同步 IO、`flow/common.py:30`
   同步函数内同步 IO——均不在 async 函数内,非 §7.2 违规。

## 架构变更(3 项结构性 P0)

### A. `agent.py` 合并 `nodes.py`

**当前结构**:
- `nodes.py`:12 个 `async def xxx_node(state, config, agent)` 函数
- `agent.py`:`KETPartnerAgent` 类,12 个方法是 `return await nodes.xxx(state, config, self)`
  的纯透传 wrapper

**问题**:`agent.py` 12 个方法是 §10.3 中间人反模式,且 `graph.py` 必须通过
`agent.xxx_node` 绑定节点,导致删除 wrapper 失去绑定。

**重构后**:
- 删除 `nodes.py`
- `KETPartnerAgent` 类内 12 个方法签名改为 `async def xxx_node(self, state, config)`
- 方法体直接调领域模块函数(`await classify_intent(self.llm_smart, ...)`)
- `graph.py` 保持 `builder.add_node("init_state", agent.init_state)`(此时
  `agent.init_state` 是真实实现方法,而非透传)
- `_run_summary_safe` / `aclose` 保留在类内
- 类规模约 275 行,单一职责(LangGraph 节点编排薄壳)

**权衡**:`CLAUDE.md §十` 严格读推荐"节点放 nodes.py",但合并入类仍符合"业务编
排薄壳"的精神——节点逻辑仍是薄壳,真正的领域计算还在 `vocab_domain` /
`sentence_domain` / `dialogue_domain`。合并入类同时消除 §10.3 中间人违规,优先级更高。

### B. `LlmService` Protocol + DI

**当前结构**(`flow/common.py:57-82`):
```python
llm_max = ChatOpenAI(...)  # 模块级硬编码实例化
llm_flash = ChatOpenAI(...)
```
业务代码 `from flow.common import llm_flash` 直接耦合,测试只能 patch 模块属性。

**重构后**(`flow/common.py`):
```python
class LlmService(Protocol):
    smart: BaseChatModel
    flash: BaseChatModel

class DashScopeLlmService:
    def __init__(self) -> None:
        api_key = _resolve_dashscope_api_key()
        self.smart = ChatOpenAI(...)   # 原 llm_max 配置
        self.flash = ChatOpenAI(...)   # 原 llm_flash 配置

default_llm_service: LlmService = DashScopeLlmService()
```

**`graph.py` 改造**:
```python
async def build_agent(
    llm_service: LlmService | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    if llm_service is None:
        llm_service = default_llm_service
    cfg = load_config()
    agent = KETPartnerAgent(llm_service, cfg)
    ...
```

**`KETPartnerAgent` 改造**:
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

**`app.py` 改造**:
```python
from flow.common import default_llm_service, logger
# ...
agent = await build_agent(default_llm_service, checkpointer=checkpointer)
```

### C. `generate_with_fallback` 三段拆分

**当前**:`sentence_domain.py:434-550`,`generate_with_fallback` 117 行,混杂造句生成、
验证重试、溢出降级、换词策略四种职责。

**重构后**:三个职责单一的辅助函数 + 主函数编排。返回值契约清晰区分"已得最终结果"与"请求外层换词重试"。

**数据契约**(替代原裸 4 元组):
```python
@dataclass(frozen=True, slots=True)
class SentenceGenerationResult:
    """最终结果。generate_with_fallback 与 _handle_overflow_fallback、
    _switch_target_or_accept 的"接受"分支都返回此类型。"""
    sentence: str
    result: ValidationResult
    target: str
    context: str

@dataclass(frozen=True, slots=True)
class _RetryOuter:
    """内部信号:_switch_target_or_accept 已切换 target,请求外层 while 循环重试。
    不对外暴露(下划线前缀)。"""
    target: str
    context: str
```

**辅助函数签名**:
```python
async def _generate_and_validate(
    llm_smart, target, context, avoid_words, avoid_sentences,
    age, profile, repos, config
) -> tuple[str, ValidationResult, list[dict]]:
    """单轮造句+验证重试循环。返回 (sentence, result, attempts)。"""

async def _handle_overflow_fallback(
    attempts: list[dict], target: str, context: str, repos
) -> SentenceGenerationResult | None:
    """若 attempts 中存在 non_ket_overflow,选 non_ket_count 最少的草稿,重跑
    validate_sentence 后返回 SentenceGenerationResult;否则返回 None 表示不适用。"""

async def _switch_target_or_accept(
    attempts, sentence, result, target, context, word_switched,
    profile, repos, config
) -> SentenceGenerationResult | _RetryOuter:
    """所有 attempts 均为 naturalness 时进入此分支:
    - word_switched 已为 True → 返回 SentenceGenerationResult(接受当前草稿)
    - word_switched 为 False 且能换到不同 target → 返回 _RetryOuter(new_target, new_context)
    - word_switched 为 False 且无其他 target 可换 → 返回 SentenceGenerationResult(接受当前草稿)
    其他 intent 分支(非全 naturalness)→ 返回 SentenceGenerationResult(接受并记 warning)"""
```

**主函数编排**:
```python
async def generate_with_fallback(
    llm_smart, initial_target, initial_context, avoid_words,
    avoid_sentences, age, profile, repos, config
) -> SentenceGenerationResult:
    target, context, word_switched = initial_target, initial_context, False
    while True:
        sentence, result, attempts = await _generate_and_validate(
            llm_smart, target, context, avoid_words, avoid_sentences,
            age, profile, repos, config
        )

        overflow = await _handle_overflow_fallback(attempts, target, context, repos)
        if overflow is not None:
            return overflow

        decision = await _switch_target_or_accept(
            attempts, sentence, result, target, context, word_switched,
            profile, repos, config
        )
        if isinstance(decision, _RetryOuter):
            target, context = decision.target, decision.context
            word_switched = True
            continue
        return decision
```

## 修复明细

### P0(8 项)

| # | 文件:行 | 修复 |
|---|---------|------|
| 1 | `state.py:5` | `KetIntent` 字面量 `"translate"` → `"translation"`,与 LLM Schema、`graph.py` 路由、`vocab_domain` 比较保持一致 |
| 2 | `vocab_domain.py:142,176,197`<br>`dialogue_domain.py:42,85,204`<br>`sentence_domain.py:186,360`<br>`agent.py:61` | 9 处 LLM 调用的 `except` 元组瘦身:每模块定义 `_LLM_RETRYABLE` 常量,只含 `(openai.APIError, asyncio.TimeoutError, pydantic.ValidationError)` 等外部失败类型,移除 `ValueError` / `AttributeError` / `TypeError` |
| 3 | `api/app.py:70, 77` | line 70 (`agent.aclose`): `except (RuntimeError, asyncio.TimeoutError)`;line 77 (`db.close`): `except (RuntimeError, OSError, aiosqlite.Error)` |
| 4 | `sentence_domain.py:444` | 定义 `@dataclass(frozen=True, slots=True) class SentenceGenerationResult`,`generate_with_fallback` 返回该类型;调用点 `nodes.py:173-184`(合并后为 `KETPartnerAgent.generate_sentence_node` 内)用命名属性访问 |
| 5 | `flow/common.py:57-82` | 见架构变更 B |
| 6 | `sentence_domain.py:552-576` | `apply_multiword_target_patch` 改纯函数:返回新 `ValidationResult`(`result.model_copy(update={...})`);调用点改 `result = apply_multiword_target_patch(target, sentence, result)` |
| 7 | `sentence_domain.py:434-550` | 见架构变更 C |
| 8 | `agent.py` + `nodes.py` | 见架构变更 A |

### P1(4 项)

| # | 文件:行 | 修复 |
|---|---------|------|
| 9 | `state.py:9-31` | 按实际写入点重写 `BTPKetState` docstring:`intent` 实际由 `classify_intent_node` 写(原写 `classify_input_node`);`profile_strategy` / `profile_weakness` 实际由 `init_state` 写(原写不存在的 `profile_summarizer_node`);`last_target_word` / `last_target_context` / `last_english_sentence` 由 `init_state` + 业务节点双写,显式枚举两个写入点 |
| 10 | 4 处 intent 分支 | `state.py` 加常量 `TRANSLATION` / `IDK` / `ASKS_MEANING` / `OFF_TOPIC` / `NON_COMPLIANT: KetIntent`;`graph.py:14-24, 33-41` 两个路由函数改用模块级 `_ROUTE_AFTER_CLASSIFY: dict[KetIntent, str]` 查表;`vocab_domain.py:208` `apply_mastery_updates` 与 `dialogue_domain.py:220` `format_output_text` 用常量替换字面量(复杂分支不强行查表) |
| 11 | `api/routes/chat.py:19-82` | 抽 `_invoke_agent(agent, req, repos, user_info, timeout) -> state` 与 `_build_chat_response(state, repos) -> ChatResponse` 两个辅助函数;主 `chat` 函数只保留编排 |
| 12 | 6 处 fallback 测试 | `tests/flow/ket_partner/test_vocab_domain.py:157-165`、`test_dialogue_domain.py:60-68,103-118,172-187`、`test_sentence_domain.py:738-746`:每个加 `bound.ainvoke.assert_awaited_once()`(异步)或 `assert_called_once()`(同步) |

### P2(10 项)

| # | 文件:行 | 修复 |
|---|---------|------|
| 13 | `cli/ket_partner/chat_logger.py` 全文 + `main.py` 调用点 | `ChatLogger` 实现 `__enter__` 返回 self,`__exit__` 调 `close_session()` 兜底;`main.py` 改 `with ChatLogger() as logger: logger.start_session(...); logger.log_turn(...)` |
| 14 | `reporting/ket_partner/exporter.py:38` | `open(out_p, "w")` 替换为 `aiofiles.open(out_p, "w")`,移除 `# noqa: ASYNC230`;若 `aiofiles` 未安装则用 `await asyncio.to_thread(...)` 包裹同步写入 |
| 15 | `persistence/bootstrap.py:51` | 把 CSV 文件读取(同步 open + DictReader)抽到同步辅助函数 `_read_csv_rows(csv_path) -> list[dict]`,在 `_import_csv` 内 `rows = await asyncio.to_thread(_read_csv_rows, csv_path)`,移除 `# noqa: ASYNC230` |
| 18 | `tests/integration/test_chat_real_llm.py:1` | 删 `import os` |
| 19 | 4 处 `state.get("intent")` | 比较前显式判 None:`vocab_domain.py:208` `apply_mastery_updates`、`dialogue_domain.py:217` `format_output_text`、`graph.py:14, 33` 两个路由函数——统一加 `intent = state.get("intent")` + `if intent is None: return <default>` |
| 20 | `dialogue_domain.py:217-243` | `format_output_text` 加 `asks_meaning` / `off_topic` / `non_compliant` 显式分支(即使返回同值也写出来),消除分支不完备嫌疑 |
| 21 | `persistence/repos.py:338` | `ProfileRepo.update` 中 `if not fields:` 分支加 `logger.warning("ProfileRepo.update called with empty fields; no-op")`,变静默为有日志 |
| 22 | `nodes.py:111` | 合并入类后(架构变更 A)此循环里的 `entry = entry.model_copy(update={"word": wr.word})` 改为直接构造新 `WrongWord` 实例,不用 `model_copy` 修改循环变量 |
| 23 | `flow/ket_partner/config.py:44-47` | JSON 加载提到模块级:`_CONFIG_DATA: Any` 在 import 时一次性加载;`load_config()` 函数体改为 `return KetConfig.model_validate(_CONFIG_DATA)`,消除 async 调用路径(`build_agent`)中的同步 IO |
| 24 | `persistence/bootstrap.py:55-61` | CSV 导入时 `word` / `pos` / `topic` / `context` 字段全部 `.strip()` 后再判空,任一必要字段缺失跳过该行(`continue`) |

## 测试策略

### 静态检查(每个 Phase 必须三项清零才能进下一 Phase)

```bash
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m ruff check .
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m mypy src
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m pytest -q
```

### 单元测试新增/补强

- **6 处 fallback 测试**(P1 #12):每条加 mock 调用次数断言。
- **`LlmService` mock 注入测试**:验证 `build_agent` 用注入的 mock `LlmService` 而非
  模块级单例,且 `KETPartnerAgent.llm_smart` / `.llm_flash` 正确读 property。
- **`KETPartnerAgent` 节点方法测试**:合并后类内方法能正确调到领域模块
  (`test_agent_nodes.py`)。
- **`SentenceGenerationResult` 字段测试**:验证 4 个命名字段、`frozen=True` 不可变、
  `slots=True` 内存占用。
- **`ChatLogger` 上下文管理器测试**:`__exit__` 在异常路径下仍调 `close_session()`
  (用 `pytest.raises` 验证)。
- **`apply_multiword_target_patch` 纯函数测试**:输入相同 `result` 多次调用,验证
  原 `result` 不被修改,返回值是新对象。
- **Intent 常量与 Schema 一致性测试**:`KetIntent` 字面量与 `IntentClassification`
  Pydantic Schema 的 `Literal` 完全一致(防回归)。

### 集成测试(§6.8 强制要求)

- **`KetIntent` 字面量对齐后**跑一次真实 LLM 调用,确认 Schema 输出的 `"translation"`
  能被新路由正确处理。标记 `@pytest.mark.integration`,skip 条件:缺
  `DASHSCOPE_API_KEY` 环境变量。
- **`LlmService` 重构后**跑一次端到端 chat 调用,确认 DI 链路通(`test_chat_real_llm.py`
  内追加用例,或新建 `test_build_agent_integration.py`)。

## 实施顺序(分 5 个 Phase)

按依赖关系排序,每个 Phase 结束 = `ruff + mypy + pytest` 三项清零 + git commit。

### Phase 1:类型契约 + 异常元组(独立、最高优先,P0 #1/#2/#3 + P2 #18)

- ① `state.py:5` `KetIntent` 字面对齐
- ② 9 处 LLM 异常元组瘦身(`vocab_domain` / `dialogue_domain` / `sentence_domain` / `agent` 各定义 `_LLM_RETRYABLE`)
- ③ `app.py:70, 77` shutdown 异常元组细化
- ④ `tests/integration/test_chat_real_llm.py:1` 删 `import os`

### Phase 2:结构性重构 LlmService + agent 合并(必须一起,P0 #5/#8)

- ⑤ `flow/common.py` 加 `LlmService` Protocol + `DashScopeLlmService` + `default_llm_service`,删除模块级 `llm_max` / `llm_flash`
- ⑥ `agent.py` 合并 `nodes.py`:12 个节点方法改真实实现;删除 `nodes.py`
- ⑦ `graph.py` `build_agent` 改 `LlmService` 参数,保持节点绑定方式不变
- ⑧ `app.py` 改 import `default_llm_service` 并传给 `build_agent`
- ⑨ **更新 `tests/integration/test_graph_integration.py`**:9 处 `from flow.ket_partner import nodes as agent_module` 改为 `from flow.ket_partner import agent as agent_module`(或 `from flow.ket_partner.agent import KETPartnerAgent` 按需调整)。验证 mock 注入点(原本 patch `nodes.xxx`)迁移到 `KETPartnerAgent.xxx_node`

### Phase 3:sentence_domain 内部(可独立,P0 #4/#6/#7)

- ⑩ 定义 `SentenceGenerationResult` dataclass
- ⑪ `apply_multiword_target_patch` 改纯函数,返回新 `ValidationResult`
- ⑫ `generate_with_fallback` 三段拆分 + 改返回类型为 `SentenceGenerationResult`
- ⑬ 调用点(`KETPartnerAgent.generate_sentence_node`,合并后)用命名属性访问

### Phase 4:P1(4 项)

- ⑭ `state.py` docstring 按实际写入点重写
- ⑮ `state.py` 加 Intent 常量,`graph.py` 改路由 dict 查表,`vocab_domain` / `dialogue_domain` 用常量
- ⑯ `api/routes/chat.py` 抽两个辅助函数
- ⑰ 6 处 fallback 测试补 `assert_awaited_once` / `assert_called_once`

### Phase 5:P2(批量收尾,8 项)

- ⑱ `chat_logger.py` 上下文管理器化 + `main.py` 调用点改造
- ⑲ `exporter.py:38` 异步 IO 替换
- ⑳ `bootstrap.py:51` CSV 读取用 `asyncio.to_thread`
- ㉑ `config.py:44-47` JSON 模块级加载
- ㉒ 4 处 `state.get("intent")` 加 None 校验
- ㉓ `format_output_text` 显式补全 Intent 分支
- ㉔ `repos.py:338` 静默 no-op 加 warning
- ㉕ `bootstrap.py:55-61` CSV 字段 strip 校验加强
- (注:#22 `nodes.py:111` 在 Phase 2 合并时自然处理,不单独列)

## 风险与权衡

### 风险

1. **Phase 2 改动面大**:`flow/common.py` / `agent.py` / `nodes.py` / `graph.py` /
   `app.py` 同时改动,中间状态可能不可运行。**对策**:Phase 2 必须在一次 commit 内
   完成,且 commit 前 `pytest` 全绿。

2. **`KetIntent` 字面量对齐可能影响存量数据**:已序列化的 state(checkpointer 中)
   如果包含 `intent` 字段,值是 `"translation"`(运行时实际值),所以从 `"translate"`
   改为 `"translation"` 对存量数据**无影响**(因为运行时从来就是 "translation")。
   但要在 spec 实施时验证一次。

3. **LLM 异常元组瘦身可能漏抓新异常**:如果 `openai` SDK 升级后新增异常类型不在
   `_LLM_RETRYABLE` 中,fallback 不会触发,异常直接逃出。**对策**:每模块的
   `_LLM_RETRYABLE` 元组以注释形式列出"为何不含 X",code review 时核对。

4. **`ChatLogger` 上下文管理器化需要改调用点**:`cli/ket_partner/main.py` 调用方式
   变化,需同步更新测试 `tests/cli/ket_partner/test_chat_logger.py` 与
   `test_main.py`。

5. **Phase 2 删除 `nodes.py` 影响 `tests/integration/test_graph_integration.py`**:
   该文件有 9 处 `from flow.ket_partner import nodes as agent_module` 用于 mock 注入
   与节点替换测试。合并后必须同步迁移到 `flow.ket_partner.agent`,且 mock 注入点
   从 `nodes.xxx` 改为 `KETPartnerAgent.xxx_node`(注意 `xxx_node` 现在是实例方法,
   patch 时要用 `patch.object(agent_instance, "xxx_node")` 或 patch 类方法)。
   **对策**:Phase 2 内同时跑 `pytest tests/integration/test_graph_integration.py`
   验证迁移完整。

### 权衡说明

- **§10 严格读推荐"节点放 nodes.py",合并入类是显式让步**:但合并同时消除 §10.3
  中间人违规,且节点仍是"业务编排薄壳"(真正领域计算在 domain 模块),符合 §10
  精神。
- **`LlmService` Protocol vs 具体类**:选 Protocol 是为了让测试 mock 时不必继承
  具体类,符合 §10.8 "服务接口必须以抽象类型暴露"。
- **不引入 Enum,只用 Literal + 常量**:Literal 保留类型安全,常量去除魔法字符串,
  与 pydantic Schema 兼容性更好(Enum 在 Pydantic v2 中需要额外配置)。

## 验收标准

每个 Phase 完成时必须满足:

1. `ruff check .` 0 错 0 警
2. `mypy src` 0 错
3. `pytest -q` 全绿(含新增测试)
4. 改动文件全部 `git add` 并 commit
5. commit message 描述本 Phase 范围与决策依据

全部 5 个 Phase 完成后,运行一次 `@pytest.mark.integration` 集成测试,确认端到端
chat 流程在新结构下正常工作。
