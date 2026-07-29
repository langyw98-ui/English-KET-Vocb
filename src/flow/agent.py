from json import dumps
from typing import (
    Any,
    Literal,
    Sequence,
    TypedDict,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.checkpoint.memory import (
    MemorySaver,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.state import (
    CompiledStateGraph,
)
from pydantic import BaseModel

from flow.common import logger

# 全局公共记忆，用于子图的状态持久化
memory = MemorySaver()


class FlowState(TypedDict):
    messages: list[AnyMessage]


class Task(BaseModel):
    method: str
    params: dict | None = None


def get_tool_call_signature(
    tool_calls: Sequence[ToolCall | dict[str, Any]],
) -> str:
    extracted = []
    for tc in tool_calls:
        args_str = dumps(
            tc.get("args", {}), sort_keys=True
        )
        extracted.append(
            f"{tc['name']}:{args_str}"
        )
    return ";".join(sorted(extracted))


# 自主智能体类
# 该类可以用于实现具有自主决策能力的智能体
class Autonomous:
    def __init__(
        self,
        model: BaseChatModel,
        prompt: str,
        tools: tuple,
    ):
        self.model = model.bind_tools(tools)
        logger.debug(prompt)
        self.system_message: list[AnyMessage] = [
            SystemMessage(content=prompt)
        ]
        self.tools = {
            tool.name: tool for tool in tools
        }
        self.hools: dict[str, Any] = {}

    async def compile(
        self,
        builder: StateGraph,
        checkpointer=None,
    ) -> CompiledStateGraph:
        builder.add_node("invoke", self.invoke)
        builder.add_node("tool", self.tool)

        builder.add_edge(START, "invoke")
        builder.add_conditional_edges(
            "invoke",
            self.condition,
            ["tool", END],
        )
        builder.add_edge("tool", "invoke")

        if checkpointer:
            return builder.compile(
                checkpointer=checkpointer
            )
        else:
            return builder.compile()

    async def invoke(self, state: FlowState):
        current_messages = list(
            state["messages"]
        )

        current_messages.append(
            await self.model.ainvoke(
                self.system_message
                + current_messages,
            )
        )

        state.update(
            {
                "messages": current_messages,
            }
        )
        # logger.info(
        #     f"\ninvoke当前消息: {state}"
        # )
        return state

    async def tool(self, state: FlowState):
        current_messages = list(
            state["messages"]
        )

        # 提取最新的用户消息，用于覆盖工具的 user_input 参数
        latest_human = None
        for msg in reversed(current_messages):
            if isinstance(msg, HumanMessage):
                latest_human = msg.content
                break

        last_msg = current_messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                # 成语接龙等需要从主图状态读取真实用户输入的工具
                # 避免主 LLM 在长对话历史中传错参数
                if (
                    latest_human is not None
                    and tool_call["name"] in self.tools
                    and "user_input" in tool_call.get("args", {})
                ):
                    tool_call["args"]["user_input"] = latest_human
                    logger.debug(
                        f"工具 {tool_call['name']} 的 user_input 已覆盖为: {latest_human}"
                    )

                tool = self.tools[tool_call["name"]]
                observation = await tool.ainvoke(
                    tool_call["args"]
                )
                if isinstance(observation, Task):
                    observation = observation.model_dump_json()
                else:
                    raise Exception(
                        f"{tool_call['name']} 工具返回值类型不是Task"
                    )
                current_messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call[
                            "id"
                        ],
                    )
                )

        state.update(
            {
                "messages": current_messages,
            }
        )

        return state

    async def condition(
        self, state: dict
    ) -> Literal["tool", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]

        if (
            not isinstance(
                last_message, AIMessage
            )
            or not last_message.tool_calls
        ):
            return "__end__"

        current_signature = (
            get_tool_call_signature(
                last_message.tool_calls
            )
        )

        for msg in reversed(messages[:-1]):
            if isinstance(msg, HumanMessage):
                break

            if (
                isinstance(msg, AIMessage)
                and msg.tool_calls
            ):
                prev_signature = (
                    get_tool_call_signature(
                        msg.tool_calls
                    )
                )

                if (
                    current_signature
                    == prev_signature
                ):
                    logger.warning(
                        f"检测到重复工具调用，强制终止: {current_signature}"
                    )
                    return "__end__"

        return "tool"


# 共识智能体类
# 该类可以用于实现需要人类共识的智能体
class Consensus(Autonomous):
    async def confirm(
        self, state: FlowState
    ): ...
    async def approval(
        self, state: FlowState
    ): ...
