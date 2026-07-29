# 代码规范合规改造 —— 第一阶段：静态类型与门禁打底详细规范

**文档日期**：2026-07-29  
**关联总方案**：[2026-07-29-code-compliance-refactoring-design.md](./2026-07-29-code-compliance-refactoring-design.md)  
**目标**：逐一定位并修复 `mypy src` 报出的 16 处静态类型错误，消除所有隐式类型漏洞，确保第一阶段门禁（Ruff + Mypy）全绿通关。

---

## 一、改造范围与问题清单

在 `mypy src` 扫描中，共检测到 16 处类型错误，分布于 2 个模块中：
1. `src/flow/agent.py`（10 处类型错误）
2. `src/flow/english_training_partner.py`（6 处类型错误）

---

## 二、`src/flow/agent.py` 问题定位与修复方案

### 问题 1.1：`hools` 属性缺失类型声明
* **代码位置**：[src/flow/agent.py:L74](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/agent.py#L74)
* **报错信息**：`error: Need type annotation for "hools"`
* **原因分析**：类构造函数中初始化 `self.hools = {}`，未标注类型，mypy 无法推导空字典的 Key/Value 类型。
* **修改方案**：
  ```python
  # 修改前
  self.hools = {}

  # 修改后
  self.hools: dict[str, Any] = {}
  ```

---

### 问题 1.2：`self.system_message` 与 `current_messages` 列表加法类型不匹配
* **代码位置**：[src/flow/agent.py:L106-L107](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/agent.py#L106-L107)
* **报错信息**：`error: Unsupported operand types for + ("list[SystemMessage]" and "list[AIMessage | HumanMessage | ChatMessage | SystemMessage | FunctionMessage | ToolMessage]")`
* **原因分析**：`self.system_message` 初始化为 `[SystemMessage(...)]`，被推导为 `list[SystemMessage]`；而 `current_messages` 包含多种消息，相加时类型不匹配。
* **修改方案**：
  在 `__init__` 中将 `self.system_message` 显式标注为 `list[BaseMessage]` 或 `list[AnyMessage]`：
  ```python
  # 修改前
  self.system_message = [
      SystemMessage(content=prompt)
  ]

  # 修改后
  self.system_message: list[BaseMessage] = [
      SystemMessage(content=prompt)
  ]
  ```

---

### 问题 1.3：Union 消息类型未收窄即访问 `.tool_calls`
* **代码位置**：[src/flow/agent.py:L133-L135](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/agent.py#L133-L135)
* **报错信息**：
  - `error: Item "HumanMessage" of ... has no attribute "tool_calls"`
  - `error: Item "ChatMessage" of ... has no attribute "tool_calls"`
  - `error: Item "SystemMessage" of ... has no attribute "tool_calls"`
  - `error: Item "FunctionMessage" of ... has no attribute "tool_calls"`
  - `error: Item "ToolMessage" of ... has no attribute "tool_calls"`
* **原因分析**：`current_messages[-1]` 的类型为 `BaseMessage`（联合类型），直接访问 `.tool_calls` 属性时，其它非 `AIMessage` 的消息子类不具备该属性。
* **修改方案**：
  在提取 `tool_calls` 前增加 `isinstance(last_message, AIMessage)` 的类型收窄判断：
  ```python
  # 修改前
  for tool_call in current_messages[-1].tool_calls:
      ...

  # 修改后
  last_msg = current_messages[-1]
  if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
      for tool_call in last_msg.tool_calls:
          ...
  ```

---

### 问题 1.4：`Literal` 装饰函数返回值的字面量参数错误
* **代码位置**：[src/flow/agent.py:L177](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/agent.py#L177)
* **报错信息**：`error: Parameter 2 of Literal[...] is invalid`
* **原因分析**：`Literal["tool", END]` 中使用变量 `END` 作为 `Literal` 参数。Mypy 要求 `Literal` 参数必须是字符串/数值字面量，而 `END` 是从 `langgraph.graph` 导入的常量字符串（实际值为 `"__end__"`）。
* **修改方案**：
  ```python
  # 修改前
  async def condition(self, state: dict) -> Literal["tool", END]:

  # 修改后
  async def condition(self, state: dict) -> Literal["tool", "__end__"]:
  ```

---

### 问题 1.5：`get_tool_call_signature` 函数入参类型注解不匹配
* **代码位置**：[src/flow/agent.py:L43-L45, L191, L205](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/agent.py#L43-L45)
* **报错信息**：`error: Argument 1 to "get_tool_call_signature" has incompatible type "list[ToolCall]"; expected "list[dict[Any, Any]]"`
* **原因分析**：`AIMessage.tool_calls` 返回类型为 `list[ToolCall]`（其中 `ToolCall` 是 `TypedDict`），而 `get_tool_call_signature` 的参数限定为 `list[dict]`。
* **修改方案**：
  更新 `get_tool_call_signature` 的入参类型标注，使其兼容 `Sequence[ToolCall | dict[str, Any]]` 或 `Sequence[Any]`：
  ```python
  # 修改前
  def get_tool_call_signature(
      tool_calls: list[dict],
  ) -> str:

  # 修改后
  def get_tool_call_signature(
      tool_calls: Sequence[ToolCall | dict[str, Any]],
  ) -> str:
  ```

---

## 三、`src/flow/english_training_partner.py` 问题定位与修复方案

### 问题 2.1：消息列表混用 `str` 字面量与 `SystemMessage`
* **代码位置**：[src/flow/english_training_partner.py:L518, L537, L539](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/english_training_partner.py#L518)
* **报错信息**：
  - `error: Argument 1 to "append" of "list" has incompatible type "SystemMessage"; expected "str"`
  - `error: Argument 1 to "extend" of "list" has incompatible type "list[...]"; expected "Iterable[str]"`
* **原因分析**：`chat_messages` 初始化时写入了裸字符串 `"你是一个智能助手..."`，被推导为 `list[str]`，后续执行 `chat_messages.append(SystemMessage(...))` 和 `chat_messages.extend(current_messages)` 时类型崩溃。
* **修改方案**：
  使用 `SystemMessage` 包装初始系统提示词，使 `chat_messages` 的类型正确初始化为 `list[BaseMessage]`：
  ```python
  # 修改前
  chat_messages = [
      "你是一个智能助手，能够友好地与用户对话。你生成的对话不能包含emoji。"
  ]

  # 修改后
  chat_messages: list[BaseMessage] = [
      SystemMessage(content="你是一个智能助手，能够友好地与用户对话。你生成的对话不能包含emoji。")
  ]
  ```

---

### 问题 2.2：`categories` 字典变量缺失类型注解
* **代码位置**：[src/flow/english_training_partner.py:L906](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/english_training_partner.py#L906)
* **报错信息**：`error: Need type annotation for "categories"`
* **原因分析**：直接声明 `categories = {}`，后续使用 `.setdefault(..., []).append(...)`，mypy 无法自动推导 Value 是 `list[str]`。
* **修改方案**：
  ```python
  # 修改前
  categories = {}

  # 修改后
  categories: dict[str, list[str]] = {}
  ```

---

### 问题 2.3：`result` 变量类型覆盖冲突与 `error_key` 属性缺失
* **代码位置**：[src/flow/english_training_partner.py:L1217-L1218](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/english_training_partner.py#L1217-L1218)
* **报错信息**：
  - `error: Incompatible types in assignment (expression has type "ErrorKeyOutput", variable has type "AnalysisOutput")`
  - `error: "AnalysisOutput" has no attribute "error_key"`
* **原因分析**：函数前半部分中 `result` 变量已被赋值并推导为 `AnalysisOutput`，在后半部分 `try:` 块中再次用 `result = await structured_llm.ainvoke(...)` 接收 `ErrorKeyOutput` 返回值，导致类型覆盖报错；后续强转 `cast(ErrorKeyOutput, result)` 未能解除已限定的变量类型绑定。
* **修改方案**：
  将提取 `ErrorKeyOutput` 的返回值绑定到独立的新变量 `ek_result`：
  ```python
  # 修改前
  result = await structured_llm.ainvoke(...)
  typed = cast(ErrorKeyOutput, result)
  error_key = typed.error_key

  # 修改后
  ek_result = await structured_llm.ainvoke(...)
  if isinstance(ek_result, ErrorKeyOutput):
      error_key = ek_result.error_key
  ```

---

### 问题 2.4：`test_cases` 变量缺失类型注解
* **代码位置**：[src/flow/english_training_partner.py:L1655](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/src/flow/english_training_partner.py#L1655)
* **报错信息**：`error: Need type annotation for "test_cases"`
* **原因分析**：模块脚手架逻辑中初始化 `test_cases = {"人工交互": {}}`，由于 Value 也是字典，mypy 无法推导内层字典结构。
* **修改方案**：
  ```python
  # 修改前
  test_cases = {"人工交互": {}}

  # 修改后
  test_cases: dict[str, dict[str, Any]] = {"人工交互": {}}
  ```

---

## 四、第一阶段验证命令 (Quality Gate)

修改完成后，在终端运行以下指令，验证第一阶段目标是否完全达成：

```bash
# 1. 语法与代码格式静态校验
ruff check .

# 2. 静态类型校验（必须 0 报错）
mypy src

# 3. 运行既有测试，确保改动未破坏现有逻辑
pytest
```
