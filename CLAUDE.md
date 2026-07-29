# 项目编码规范

本规范均为通用编程原则，不绑定具体业务。写代码与 review 时按此对照。

## 一、异常处理纪律

1. 禁止裸 `except Exception` / 裸 `except`，必须捕获具体异常类型（如 `ValidationError`、`KeyError`、`AttributeError`、`OSError` 等的显式元组）。裸 except 会把字段名写错这类代码 bug 当成 LLM/IO 失败一起吞掉。
2. 所有跨边界调用（LLM、HTTP、文件、子进程、外部服务）必须有 `try/except` 保护，缺一不可。
3. 所有 fallback 分支必须用 `logger.warning(..., exc_info=True)` 留下日志痕迹，不允许静默兜底。
4. 拿到结构化 / 外部返回值后，即使"成功"也必须校验非空、类型正确，再使用。
5. 跨边界 `try/except` 的异常元组必须严格区分"外部失败"与"代码 bug"——只按"具体外部失败类型"枚举（如 `ValidationError`、`OSError`、`TimeoutError`、外部 SDK 自定义异常）；**禁止包含可能由代码 bug 引发的通用异常**（`ValueError` / `TypeError` / `KeyError` / `AttributeError` / `IndexError` 等）——这类异常必须直接暴露被测试捕获，不允许被当成外部失败静默兜底。**例外**：在边界适配层（Parser / Adapter / 数据清洗函数）内，允许捕获 `ValueError` 等标准库异常并重新封装为业务定义的具体失败类型（如 `DataFormatError`）；封装后业务层禁止再捕获 `ValueError`。
6. Fallback 测试既要验证"兜底返回值正确"，也要验证"非兜底路径在正常输入下能产出正确返回值"。fallback 默认值与断言期望值重合时，永远走 fallback 也能通过测试，无法捕获 bug。

## 二、类型与契约正确性

1. 类型注解必须与运行时实际类型一致（例：实际装 `BaseMessage` 就不能写 `list[str]`）。
2. 取值集合有限的字段（状态字段、分类标签、路由结果、错误码）必须用类型系统约束（枚举 / `Literal` / 常量集合），让非法值在赋值或调用阶段直接失败。开放类型（`str` / `int`）仅用于真正开放取值（用户输入、外部自由文本等）。状态对象的字段是高发区，禁止为"灵活"而退化为开放类型。
3. 比较操作必须类型匹配（不能拿 `dict` 比 `str`、`None` 比字符串、列表比标量）。
4. 引用属性 / 键 / 索引前必须确认其存在；不依赖"可能存在"的幻觉字段或方法。
5. 框架与外部 API 的契约键名必须严格核对（键名 typo 是静默失败的高发原因，例如 LangGraph 状态更新字典的键名写错会被静默丢弃）。
6. 本项目中所有的 LLM 调用必须统一使用结构化输出的形式（`with_structured_output` + Pydantic Schema），禁止依赖裸文本或正则解析（参考 `aimo/agents/study_english_speaking_buddy` 的所有 LLM 调用实现）。
7. 方法重写（`@override`）必须遵守 Liskov 替换原则：参数类型、返回类型、异常声明必须与基类兼容，子类不得收窄参数类型。同一基类的多个子类应保持一致的 override 签名范式，新增子类前先查阅兄弟实现的签名。
8. `ClassVar` 注解只能用在类体内赋值；模块级常量禁用 `ClassVar`，直接用普通类型注解（如 `_CONST: dict[str, str] = {...}`）。
9. `cast("TypeName", ...)` 的字符串前向引用必须先 `import TypeName`。`from __future__ import annotations` 只推迟**注解**的运行时解析，**不影响函数调用的字符串参数**（`cast` 的第一个参数是字符串参数，不在注解延迟范围内）；ruff F821 / mypy name-defined 仍要求该名字在作用域内可解析。示例：`cast("RunnableConfig", {...})` 必须先 `if TYPE_CHECKING: from langchain_core.runnables import RunnableConfig`。
10. 跨边界接收的字符串（外部输入、API 响应字段、生成器输出）必须显式校验非空白（去空白后非空），不依赖框架默认行为或调用方契约。空字符串在多数类型校验中会通过，但业务上几乎总是非法。
11. 跨函数 / 跨模块边界禁用裸多元组，必须用命名字段类型（`dataclass` / `record` / `struct` / 类 / 命名元组）——调用方按位置取值时位置错位运行时不会报错。值对象优先不可变（`frozen=True` / `slots=True` 或等价语义）。**模块内部私有方法 / 闭包允许返回多元组，但调用方必须使用命名解包（`a, b = func()`），禁止索引访问（`func()[0]`）**。


## 三、状态与可变性

1. 优先纯函数；不修改入参，返回新对象。函数的副作用只通过返回值体现，不通过偷偷改入参体现。
2. 共享可变状态必须有明确的单一写入者（single-writer），其他位置只读。
3. 共享可变状态的字段必须在所属类型的 docstring 里逐字段声明写入者，格式：
   ```
   - field_name: 仅 <写入者> 在 <条件> 时写；<其他位置> 只读
   ```
   多写入者必须显式枚举每个写入点。

## 四、代码组织

1. 函数单一职责；超过约 80 行或包含 3 个以上并列分支的函数必须按职责拆分。
2. 重复逻辑出现 2 次以上必须抽取为命名辅助函数，禁止复制粘贴式复用（复制粘贴是字段名写错等隐性 bug 的高发来源）。
3. 魔法数字、魔法字符串必须命名为常量（模块级或类级 `ClassVar`）。同类并列的映射（错误码→文案、分类→回复模板、标签→处理逻辑）必须封装为模块级命名常量（字典 / 查表）；禁止在业务函数里内联 `if/elif` 字面量分支或硬编码映射表。
4. 数据契约（pydantic / TypedDict / Schema）与业务逻辑必须分模块，不在逻辑文件里内嵌大量类型定义。
5. 同类的并列项（节点集合、键集合、错误码表、配置项）封装为命名常量，不写内联元组或硬编码列表。

## 五、领域建模完备性

1. 决策 / 分类的取值集合必须覆盖边界与中间态（如 `uncertain` / `unknown`），避免二元判断漏掉真实状态。
2. 每条分支都必须产出可用结果，禁止 dead path 或"什么都没发生"的返回。

## 六、测试

1. 测试必须覆盖正常路径 + 兜底路径（fallback 也要测，且至少测一次）。
2. 框架契约（序列化、配置加载、状态合并、Schema 校验）必须有专门测试，不只测业务逻辑。
3. 测试文件必须按"单一被测单元"拆分。被测单元指：一个函数、一个类、一个模块、一组紧密耦合的规则。命名必须反映被测对象（`test_<单元名>_<场景>.py`）。禁止一个测试文件覆盖多个无关联单元。
4. 当被测代码的兜底返回值与正常路径返回值类型 / 形态重合时，仅断言返回值无法区分两条路径——被测函数内部异常导致从未调用 mock、直接走兜底时，测试仍会通过。此类测试必须额外断言 mock 被以预期参数调用过（同步 `call_count` / `assert_called*`；异步 `await_count` / `assert_awaited*`）。即使返回值形态不重合，凡是用 `unittest.mock.patch` 替换目标的测试也必须做此断言，不能只断言返回值。
5. Mock API 必须匹配目标类型：
   - 同步方法用 `MagicMock`，断言用 `assert_called*` / `call_count`
   - 异步方法或返回 `AsyncMock` 的同步方法用 `AsyncMock`，断言用 `assert_awaited*` / `await_count`
   - `MagicMock` 无 `assert_not_awaited`；该断言只存在于 `AsyncMock`
   - `side_effect=[a, b, ...]` 列表长度必须 ≥ 实际调用次数，否则第 N 次迭代抛 `StopIteration`
   - `side_effect` 模式下 `.return_value` 是默认 `MagicMock`，不会被实际调用填充——断言调用次数请用 `call_count` / `await_count`，不要访问 `.return_value.<counter>`
6. 单元测试必须 hermetic（可重复、不依赖外部环境）：禁止真实调用 LLM API / HTTP / 数据库 / 文件系统外部路径；必须跨边界时，要么 patch 边界，要么用输入短路让被测代码走不触发边界的分支。集成测试若必须调用真实外部服务，必须标记 `@pytest.mark.integration` 并配 skip 条件（如缺 API key 时跳过）。
7. 禁止使用未在 `pyproject.toml` / `requirements.txt` 显式声明的测试装饰器或插件。pytest 配置必须开启 `filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]`，把"未注册 mark"作为错误，防止装饰器形同虚设。
8. LLM 结构化输出 Schema 变更（新增 / 删除 / 改名字段）必须同步三件事：① 更新对应 Prompt 模板；② 新增 / 修改对应字段的断言测试用例；③ 运行一次真实 LLM 调用验证（集成测试标记 `@pytest.mark.integration` 并配 skip 条件）。仅有 mock 测试不算完成——mock 永远返回"合法结构"，无法发现 Prompt 实际不产出该字段的问题。

## 七、资源与异步

1. 资源（文件、锁、连接、子进程、临时对象等）必须用 `with` 或 `try/finally` 显式释放，禁止依赖 GC 回收。
2. 异步函数内禁止调用同步阻塞 IO（`requests`、`time.sleep`、同步文件读写、阻塞 socket 等），必须替换为异步等价物（`aiohttp`、`asyncio.sleep`、`aiofiles` 等）；不可避免的 CPU 密集计算用 `asyncio.to_thread` / `run_in_executor` 卸载到线程池。
3. `CancelledError` 只能在 `finally` 中做清理，禁止在 `except` 中捕获后丢弃；如确需捕获必须重新抛出。禁止 `except BaseException` 兜底，避免误吞取消信号导致任务僵尸。

## 八、Brief / Plan 编写规范

1. Brief 内部必须自洽——impl 代码块、测试断言、引用的既有常量值（字符串字面、数值、枚举名）三者必须字符级一致。任何一处出现歧义（如既有常量值是 "X" 但测试断言含 "Y"）都视为 brief 缺陷，必须在派发前修正。
2. Brief 引用既有符号（模块常量、类方法、基类字段）时必须显式注明来源（如"Task N 已创建" / "模块级常量" / "基类 `BaseAgentNodes` 提供"），不能让 implementer 自己猜测是否需要新建。
3. Brief 中每个测试用例必须明确四项：被 mock 的目标、mock 的类型（同步/异步）、预期调用次数、对 mock 的断言。缺一项都会导致测试流于形式。

## 九、静态检查纪律

1. 每个 task 的"verify pass"步骤必须包含三项：`ruff check`、`mypy`、`pytest`，三项全部清零才能提交。只跑 pytest 不算通过。
2. CI 或本地配置必须把静态检查的 warning 视为 error：ruff 默认严格；mypy 开启 `strict = true` 或关键检查（`disallow_untyped_defs`、`warn_return_any`、`no_implicit_optional`）。
3. 所有 task 累积的 deferred Minor（轻微问题）必须在合并到主干前一次性清理。禁止把静态检查错误、未使用 import、类型注解缺失等带到主干。
4. 第三方库无类型注解时，必须编写 Wrapper 模块封装调用，`# type: ignore` 仅允许出现在 Wrapper 内部，且必须附注释说明原因（如 `# type: ignore[no-untyped-def]  # upstream SDK untyped`）。业务代码禁止用 `# type: ignore` 掩盖类型问题。

## 十、模块分层与依赖方向

1. 包含 3 种以上正交职责（业务编排 / 外部调用 / 决策路由）的模块必须按职责拆分为独立文件，每个文件单一职责。单一文件混合多种正交职责是变更爆炸半径失控的高发原因。

   **LangGraph 项目的三层映射**：
   - `nodes.py`：集中定义所有 LangGraph 节点函数（仅做 state 读写的薄壳，属"业务编排"层）
   - `graph.py`：图拓扑编排 + `route_*` 条件函数（属"决策路由"层）
   - `scene.py` / `negotiation.py` 等领域模块：跨边界 LLM/IO 调用与领域计算（属"外部调用"层）

   三层不可混合在同一文件。以下两类常见反模式均违反本条：
   - **命名与内容错位**：文件名为 `nodes.py` 但实际装的是非 node helper（被节点调用的业务函数），真正的节点定义散落在 `agent.py` 的类方法里——文件名必须如实反映其单一职责。
   - **按领域拆节点**：把节点和它的业务 helper 一起塞进 `nodes/negotiation.py` 之类的领域文件——节点是图抽象的顶点，不归任何单一领域，按领域拆节点会混淆图层与领域层。
2. 模块间依赖方向必须单向无环（DAG），下层模块不感知上层存在。禁止循环 import 或运行时反向调用。
3. 跨边界调用（外部服务、IO、子系统、第三方 SDK）必须封装为独立的服务 / 客户端模块，业务逻辑不得直接耦合具体 SDK。服务接口必须以抽象类型暴露（如 `Protocol` / 接口类 / 基类），便于替换实现。
4. 服务模块必须支持依赖注入——构造函数接受可选的服务实例参数，默认值用具体实现。这样单元测试可注入 mock，避免触发真实外部调用。禁止业务模块在内部 `new` 跨边界依赖实例。
