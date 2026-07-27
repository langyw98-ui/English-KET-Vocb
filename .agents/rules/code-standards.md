# 项目编码规范

本规范均为通用编程原则，不绑定具体业务。写代码与 review 时按此对照。

## 一、异常处理纪律

1. 禁止裸 `except Exception` / 裸 `except`，必须捕获具体异常类型（如 `ValidationError`、`KeyError`、`AttributeError`、`OSError` 等的显式元组）。裸 except 会把字段名写错这类代码 bug 当成 LLM/IO 失败一起吞掉。
2. 所有跨边界调用（LLM、HTTP、文件、子进程、外部服务）必须有 `try/except` 保护，缺一不可。
3. 所有 fallback 分支必须用 `logger.warning(..., exc_info=True)` 留下日志痕迹，不允许静默兜底。
4. 拿到结构化 / 外部返回值后，即使"成功"也必须校验非空、类型正确，再使用。
5. 跨边界 `try/except` 的异常元组必须严格区分"外部失败"与"代码 bug"——只捕获外部失败（如 `ValidationError`、`OSError`、`TimeoutError`、外部 SDK 自定义异常）；禁止把 `KeyError` / `AttributeError` / `TypeError` 等可能由字段名写错、属性误用、类型不匹配引起的代码 bug 类异常放入跨边界捕获元组。这类异常必须直接暴露，不允许被当成外部失败静默兜底。
6. Fallback 测试既要验证"兜底返回值正确"，也要验证"非兜底路径在正常输入下能产出正确返回值"。fallback 默认值与断言期望值重合时，永远走 fallback 也能通过测试，无法捕获 bug。

## 二、类型与契约正确性

1. 类型注解必须与运行时实际类型一致（例：实际装 `BaseMessage` 就不能写 `list[str]`）。
2. 闭集取值用 `Literal` / 枚举约束，让非法值在校验阶段失败，而不是流到业务里猜。
3. 比较操作必须类型匹配（不能拿 `dict` 比 `str`、`None` 比字符串、列表比标量）。
4. 引用属性 / 键 / 索引前必须确认其存在；不依赖"可能存在"的幻觉字段或方法。
5. 框架与外部 API 的契约键名必须严格核对（键名 typo 是静默失败的高发原因，例如 LangGraph 状态更新字典的键名写错会被静默丢弃）。
6. 本项目中所有的 LLM 调用必须统一使用结构化输出的形式（`with_structured_output` + Pydantic Schema），禁止依赖裸文本或正则解析（参考 `aimo/agents/study_english_speaking_buddy` 的所有 LLM 调用实现）。
7. 方法重写（`@override`）必须遵守 Liskov 替换原则：参数类型、返回类型、异常声明必须与基类兼容，子类不得收窄参数类型。同一基类的多个子类应保持一致的 override 签名范式，新增子类前先查阅兄弟实现的签名。
8. `ClassVar` 注解只能用在类体内赋值；模块级常量禁用 `ClassVar`，直接用普通类型注解（如 `_CONST: dict[str, str] = {...}`）。
9. `cast("TypeName", ...)` 的字符串前向引用必须先 `import TypeName`。`from __future__ import annotations` 只推迟运行时解析，ruff F821 / mypy name-defined 仍要求该名字在作用域内可解析。


## 三、状态与可变性

1. 优先纯函数；不修改入参，返回新对象。函数的副作用只通过返回值体现，不通过偷偷改入参体现。
2. 共享可变状态必须有明确的单一写入者（single-writer），其他位置只读。
3. 共享状态的写入规则必须在注释中显式声明"谁写、何时写"，避免隐式契约。

## 四、代码组织

1. 函数单一职责；超过约 80 行或包含 3 个以上并列分支的函数必须按职责拆分。
2. 重复逻辑出现 2 次以上必须抽取为命名辅助函数，禁止复制粘贴式复用（复制粘贴是字段名写错等隐性 bug 的高发来源）。
3. 魔法数字、魔法字符串必须命名为常量（模块级或类级 `ClassVar`）。
4. 数据契约（pydantic / TypedDict / Schema）与业务逻辑必须分模块，不在逻辑文件里内嵌大量类型定义。
5. 同类的并列项（节点集合、键集合、错误码表、配置项）封装为命名常量，不写内联元组或硬编码列表。

## 五、领域建模完备性

1. 决策 / 分类的取值集合必须覆盖边界与中间态（如 `uncertain` / `unknown`），避免二元判断漏掉真实状态。
2. 每条分支都必须产出可用结果，禁止 dead path 或"什么都没发生"的返回。

## 六、测试

1. 测试必须覆盖正常路径 + 兜底路径（fallback 也要测，且至少测一次）。
2. 框架契约（序列化、配置加载、状态合并、Schema 校验）必须有专门测试，不只测业务逻辑。
3. 测试文件按组件 / 节点拆分，禁止一个测试文件涵盖过多职责。
4. 每个用 `unittest.mock.patch` 替换目标的测试，必须断言 mock 被以预期参数调用（如 `mock.assert_awaited_once()` / `mock.assert_called_with(...)` / `mock.call_count == N`），不能只断言被测函数的返回值。否则被测函数内部异常导致 mock 从未被调用、走兜底路径时，只要兜底返回值恰好满足断言，测试也会通过。
5. Mock API 必须匹配目标类型：
   - 同步方法用 `MagicMock`，断言用 `assert_called*` / `call_count`
   - 异步方法或返回 `AsyncMock` 的同步方法用 `AsyncMock`，断言用 `assert_awaited*` / `await_count`
   - `MagicMock` 无 `assert_not_awaited`；该断言只存在于 `AsyncMock`
   - `side_effect=[a, b, ...]` 列表长度必须 ≥ 实际调用次数，否则第 N 次迭代抛 `StopIteration`
   - `side_effect` 模式下 `.return_value` 是默认 `MagicMock`，不会被实际调用填充——断言调用次数请用 `call_count` / `await_count`，不要访问 `.return_value.<counter>`
6. 单元测试必须 hermetic（可重复、不依赖外部环境）：禁止真实调用 LLM API / HTTP / 数据库 / 文件系统外部路径；必须跨边界时，要么 patch 边界，要么用输入短路让被测代码走不触发边界的分支。集成测试若必须调用真实外部服务，必须标记 `@pytest.mark.integration` 并配 skip 条件（如缺 API key 时跳过）。
7. 禁止使用未在 `pyproject.toml` / `requirements.txt` 显式声明的测试装饰器或插件。pytest 配置必须开启 `filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]`，把"未注册 mark"作为错误，防止装饰器形同虚设。

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
