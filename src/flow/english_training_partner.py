"""
英语场景练习智能体 — 面向5~12岁中文母语儿童的英语口语陪练。
"""

import asyncio
import json
import random
import re
from inspect import currentframe
from os.path import dirname, realpath
from typing import Literal, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from flow.agent import Task, memory
from flow.common import logger

# ── State ────────────────────────────────────────────────────────────


class BTPState(TypedDict):
    messages: list[AnyMessage]
    intent: str | None  # "tool" | "flow_resp"
    next_action_node: str | None  # 下一轮应进入的节点
    phase: str | None  # "negotiation" | "dialogue"
    mode: str | None  # "scene" | "free_chat"
    scene_category: str | None  # 场景类别
    scene_name: str | None  # 场景名称
    ai_role: str | None  # AI 扮演的角色
    user_role: str | None  # 用户扮演的角色
    proposed_scene: dict | None  # 当前提议的场景信息
    input_type: str | None  # "non_compliant" | "vocab_gap" | "sentence_gap" | "grammar_error" | "clean"
    teach_words: str | None  # 需要教的词汇（中文→英文映射）
    correct_sentence: str | None  # 修正后的完整句子
    error_explain: str | None  # 错误解释（中文，简洁）
    error_counts: dict | None  # 同类语法错误纠正次数追踪
    awaiting_scene_selection: bool | None  # 上轮展示了场景列表，等待用户选择
    valid_dialogues: list[AnyMessage]  # 有效对话记录（clean轮次+开场白）
    tool_result: str | None


# ── Structured Output Models ──────────────────────────────────────────


class AnalysisOutput(BaseModel):
    teach_words: str = Field(
        description='必须选出至少1个需要教的词汇，格式"中文词"对应的英文是"English word"，多个用逗号分隔'
    )
    correct_sentence: str = Field(description="修正后的完整英文句子，必须填写")
    error_explain: str | None = Field(
        default=None, description="简短的中文错误解释，无错误时填null"
    )


class ErrorKeyOutput(BaseModel):
    error_key: str = Field(
        description="标准化错误标识符，如 vocab:umbrella 或 grammar:past_tense"
    )


class CustomSceneOutput(BaseModel):
    name: str = Field(description="场景名称，简短中文，如「聊宠物」「在公园玩」")
    category: str = Field(description="场景类别，如「兴趣」「日常」「想象」")
    ai_role: str = Field(description="AI扮演的角色，英文，如「friend」「doctor」")
    user_role: str = Field(description="用户扮演的角色，英文，如「friend」「patient」")
    reference_opening: str = Field(description="AI的英文开场白，简单自然的一句话")


class NegotiationIntent(BaseModel):
    intent: Literal[
        "accept",
        "select_scene",
        "custom_scene",
        "custom_request",
        "free_chat",
        "reroll",
        "list_scenes",
        "non_compliant",
    ] = Field(description="用户意图分类标签")
    confidence: float = Field(description="分类置信度，取值0.0到1.0")


# ── Tool ─────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
def exit_english_training_partner() -> Task:
    """主动退出"英语场景练习模式"模式
    用户的输入中必须显式包含"退出"或"结束"这两个关键词时，才能调用此工具。
    """
    _frame = currentframe()
    assert _frame is not None

    return Task.model_validate(
        {
            "method": _frame.f_code.co_name,
            "params": {
                "expect_reply": True,
                "reply_content": "已退出英语场景练习模式。",
            },
        }
    )


# ── Hardcoded fallback scenes ───────────────────────────────────────

_FALLBACK_SCENES = [
    {
        "name": "点冰淇淋",
        "category": "餐饮",
        "ai_role": "店员",
        "user_role": "客人",
        "reference_opening": "Hi! Welcome to our ice cream shop. What flavor would you like?",
    },
    {
        "name": "问路",
        "category": "求助",
        "ai_role": "路人",
        "user_role": "问路者",
        "reference_opening": "Excuse me, are you looking for a place? I might be able to help.",
    },
    {
        "name": "聊宠物",
        "category": "兴趣",
        "ai_role": "朋友",
        "user_role": "朋友",
        "reference_opening": "I love animals! Do you have any pets at home?",
    },
]


# ── Agent ────────────────────────────────────────────────────────────


class BTPAutonomous:
    """双语场景练习智能体"""

    def __init__(
        self,
        model: BaseChatModel,
        model_smart: BaseChatModel,
        tools: list,
        info: dict,
        system_prompt: str,
    ):
        self.model_with_tools = model.bind_tools(tools)
        self.original_model = model
        self.original_model_smart = model_smart
        self.tools = {t.name: t for t in tools}
        self.system_prompt = SystemMessage(content=system_prompt)
        self.info = info
        self.is_init_state = True
        self.scenes: list[dict] = []
        self._last_proposed_name: str | None = None

    # ── compile ──────────────────────────────────────────────────

    async def compile(
        self,
        builder: StateGraph,
        checkpointer,
    ) -> CompiledStateGraph:
        builder.add_node("init_state", self.init_state)
        builder.add_node("intent_classifier", self.intent_classifier)
        builder.add_node("tool_executor", self.tool_executor)
        builder.add_node("set_tool_call_hint", self.set_tool_call_hint)
        builder.add_node("flow_resp", self.flow_resp)
        builder.add_node("exit_in_reason", self.exit_in_reason)
        builder.add_node("negotiation_handler", self.negotiation_handler)
        builder.add_node("input_classifier", self.input_classifier)
        builder.add_node("teach_vocab", self.teach_vocab)
        builder.add_node("teach_sentence", self.teach_sentence)
        builder.add_node("teach_grammar", self.teach_grammar)
        builder.add_node("compliance_redirect", self.compliance_redirect)
        builder.add_node("dialogue_respond", self.dialogue_respond)
        builder.add_node("explain_meaning", self.explain_meaning)

        # Edges
        builder.add_edge(START, "init_state")
        builder.add_edge("init_state", "intent_classifier")
        builder.add_conditional_edges(
            "intent_classifier",
            self.route_intent,
            {"tool": "tool_executor", "flow_resp": "flow_resp"},
        )
        builder.add_edge("tool_executor", "set_tool_call_hint")
        builder.add_edge("set_tool_call_hint", END)
        builder.add_conditional_edges(
            "flow_resp",
            self.route_by_phase,
            ["negotiation_handler", "input_classifier", "exit_in_reason"],
        )

        # Negotiation → END
        builder.add_edge("negotiation_handler", END)

        # exit_in_reason → END
        builder.add_edge("exit_in_reason", END)

        # Input classification routing
        builder.add_conditional_edges(
            "input_classifier",
            self.route_input_type,
            [
                "compliance_redirect",
                "teach_vocab",
                "teach_sentence",
                "teach_grammar",
                "explain_meaning",
                "dialogue_respond",
            ],
        )

        # Teach nodes → END (teaching content is the final reply, wait for user input)
        builder.add_edge("teach_vocab", END)
        builder.add_edge("teach_sentence", END)
        builder.add_edge("teach_grammar", END)

        # compliance_redirect → END
        builder.add_edge("compliance_redirect", END)

        # dialogue_respond → END
        builder.add_edge("dialogue_respond", END)

        # explain_meaning → END
        builder.add_edge("explain_meaning", END)

        return builder.compile(checkpointer=checkpointer)

    # ── init_state ───────────────────────────────────────────────

    async def init_state(self, state: BTPState) -> BTPState:
        logger.debug("【NODE】init_state")
        if self.is_init_state:
            state["intent"] = None
            state["next_action_node"] = None
            state["phase"] = "negotiation"
            state["mode"] = None
            state["scene_category"] = None
            state["scene_name"] = None
            state["ai_role"] = None
            state["user_role"] = None
            state["proposed_scene"] = None
            state["input_type"] = None
            state["teach_words"] = None
            state["correct_sentence"] = None
            state["error_explain"] = None
            state["error_counts"] = {}
            state["awaiting_scene_selection"] = False
            state["valid_dialogues"] = []
            state["tool_result"] = None
            self.is_init_state = False

            # Load scene library
            self.scenes = self._load_scenes()
        else:
            valid = state.get("valid_dialogues") or []
            if valid:
                logger.debug(f"{valid[-5:]}")
        return state

    def _load_scenes(self) -> list[dict]:
        """Load scenes from data/scenes.json; fall back to hardcoded list."""
        try:
            scene_path = f"{realpath(dirname(__file__))}/data/scenes.json"
            with open(scene_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            flat = []
            for cat in data.get("categories", []):
                for s in cat.get("scenes", []):
                    flat.append({**s, "category": cat["name"]})
            return flat if flat else _FALLBACK_SCENES
        except Exception as e:
            logger.warning(f"场景库加载失败，使用兜底场景: {e}")
            return list(_FALLBACK_SCENES)

    async def _generate_scene_desc(self, scene: dict) -> str:
        """让 LLM 根据场景资料生成一句自然的场景描述。"""
        prompt = [
            SystemMessage(
                content=(
                    "你正在向5~12岁的孩子推荐一个英语角色扮演场景。\n"
                    "请根据下面的场景资料，用中文写一句简短的陈述句进行场景描述（不超过30字）。\n"
                    "【要求】：\n"
                    "- 不要使用模板格式，不要出现「我扮演X，你扮演Y」这种生硬表述\n"
                    "- 自然地融入角色关系，让孩子能理解当前的场景\n"
                    "- 只输出一句话，不要有任何其他文字"
                    "【场景描述示例：】1.你扮演来文具店购买文具的人。2.你扮演来参加朋友的生日聚会的人。3.你扮演在运动会上遇到了一个朋友。\n"
                )
            ),
            HumanMessage(
                content=(
                    f"场景类别：{scene.get('category', '')}\n"
                    f"场景名称：{scene.get('name', '')}\n"
                    f"AI角色：{scene.get('ai_role', '')}\n"
                    f"用户角色：{scene.get('user_role', '')}\n"
                )
            ),
        ]
        try:
            resp = await self.original_model.ainvoke(prompt)
            return (
                f"我推荐的场景是一个{scene.get('category', '')}类场景："
                + cast(str,resp.content).strip()
            )
        except Exception:
            return f"{scene['category']}中的{scene['name']}场景"

    async def _generate_custom_scene(self, user_desc: str) -> dict | None:
        """根据用户描述生成自定义场景字段。"""
        prompt = [
            SystemMessage(
                content=(
                    "用户想玩一个英语角色扮演场景，但描述的场景不在预设列表中。"
                    "请根据用户的描述，生成一个适合5~12岁儿童的角色扮演场景。\n"
                    "以JSON格式输出。\n"
                    "要求：\n"
                    "- name：场景名称，简短中文（如「在游乐场玩」）\n"
                    "- category：场景类别，中文（如「兴趣」「日常」「想象」「校园」）\n"
                    "- ai_role：AI扮演的角色，中文（如「friend」）\n"
                    "- user_role：用户扮演的角色，中文（如「friend」）\n"
                    "- reference_opening：AI的英文开场白，简单自然的一句话\n"
                    "- 必须输出所有5个字段，缺一不可\n"
                    "- 内容必须适合儿童"
                )
            ),
            HumanMessage(content=f"用户描述：{user_desc}"),
        ]
        try:
            structured_llm = self.original_model.with_structured_output(
                CustomSceneOutput, method="function_calling"
            )
            result = await structured_llm.ainvoke(prompt)
            if isinstance(result, dict):
                return result
            return result.model_dump()
        except Exception as e:
            logger.warning(f"自定义场景生成失败: {e}")
            return None

    # ── intent_classifier ────────────────────────────────────────

    async def intent_classifier(self, state: BTPState) -> BTPState:
        """
        非action节点
        仅用于init_state节点后的意图识别，因为这个情况下无需设置interrupt
        意图识别器：判断用户输入属于"功能需求"（退出留言模式），还是继续留言
        """
        logger.debug("【NODE】intent_classifier")
        current_messages = list(state["messages"])
        logger.debug(f"当前消息列表: {current_messages}")
        # 获取用户消息
        user_message = current_messages[-1].content if current_messages else ""
        # 构建意图识别的提示词
        intent_prompt = [
            SystemMessage(
                content="""
                    你是一个意图分类器。请分析用户的输入，判断其意图类别。

                    判断规则：
                    - 如果用户要求执行:退出“英语陪练”模式，返回 "tool"
                    - 如果用户回复的内容是“确认某项操作命令”或与“练习英语对话”相关的操作命令，返回 "flow_resp"

                    特别注意：
                    - 如果用户明确要求"退出“英语陪练”模式"，返回 "tool"

                    输出格式：只输出一个单词，要么是 "tool"，要么是 "flow_resp"，不要有任何其他文字。
                """
            ),
            HumanMessage(content=user_message),
        ]

        # 调用模型进行意图识别
        try:
            response = await self.model_with_tools.ainvoke(intent_prompt)
            # 检查是否有工具调用
            if hasattr(response, "tool_calls") and response.tool_calls:
                # 模型尝试调用工具，说明是工具意图
                intent = "tool"
            else:
                # 否则从内容中解析
                intent = cast(str,response.content).strip().lower() if response.content else ""
                # 清理可能的额外内容
                intent = intent.replace(" ", "").replace("\n", "")
            # 验证意图结果
            if intent not in ["tool", "flow_resp"]:
                logger.warning(f"无法识别的意图: '{intent}', 默认为 flow_resp")
                intent = "flow_resp"

        except Exception as e:
            logger.error(f"意图识别失败: {e}, 默认为 flow_resp")
            intent = "flow_resp"

        state["messages"] = current_messages
        state["intent"] = intent
        return state

    # ── route_intent ─────────────────────────────────────────────

    async def route_intent(self, state: BTPState) -> Literal["tool", "flow_resp"]:
        return "tool" if state.get("intent") == "tool" else "flow_resp"

    # ── tool_executor ────────────────────────────────────────────

    async def tool_executor(self, state: BTPState) -> BTPState:
        """
        非action节点
        工具执行器：调用相应的外部工具
        """
        logger.debug("【NODE】tool_executor")
        current_messages = list(state["messages"])

        # 使用绑定了工具的模型，让模型自动选择并调用工具
        response = await self.model_with_tools.ainvoke(current_messages)

        # 添加AI响应到消息历史
        current_messages.append(response)
        state["messages"] = current_messages

        # 如果模型调用了工具，执行工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    try:
                        result = await tool.ainvoke(tool_args)
                        state["tool_result"] = str(result)
                        logger.info(f"工具 {tool_name} 执行成功")

                        # 添加工具执行结果到消息历史
                        current_messages.append(
                            ToolMessage(
                                content=result.model_dump_json(),
                                tool_call_id=tool_call["id"],
                            )
                        )
                    except Exception as e:
                        state["tool_result"] = f"工具执行失败: {e!s}"
                        logger.error(f"工具 {tool_name} 执行失败: {e}")
                        current_messages.append(
                            ToolMessage(
                                content=f"工具执行失败: {e!s}",
                                tool_call_id=tool_call["id"],
                            )
                        )
                else:
                    state["tool_result"] = f"未找到工具: {tool_name}"
                    logger.warning(f"未找到工具: {tool_name}")
                    current_messages.append(
                        ToolMessage(
                            content=f"未找到工具: {tool_name}",
                            tool_call_id=tool_call["id"],
                        )
                    )

        state["messages"] = current_messages
        return state

    async def set_tool_call_hint(self, state: BTPState) -> BTPState:
        """
        大模型聊天：常规对话模式
        """
        current_messages = list(state["messages"])
        # 获取记忆内容
        memory_content = state.get("memory", "")

        # 检查工具结果中是否有 expect_reply 和 reply_content 字段
        reply_content = None
        for msg in reversed(current_messages):
            if msg.type == "tool":
                # 尝试解析工具结果中的 expect_reply 和 reply_content
                try:
                    import json

                    result = (
                        json.loads(msg.content)
                        if isinstance(msg.content, str)
                        else msg.content
                    )
                    if isinstance(result, dict):
                        params = result.get("params", {})
                        if params.get("expect_reply"):
                            reply_content = params.get("reply_content")
                            logger.info(f"工具期望生成回复，预设内容: {reply_content}")
                            break
                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    pass

        # 如果有预设的回复内容，直接使用
        if reply_content:
            current_messages.append(AIMessage(content=reply_content))
            logger.info(f"使用预设回复: {reply_content}")
            state["messages"] = current_messages
            return state

        # 构建聊天消息，使用原始模型（未绑定工具）
        chat_messages = [
            "你是一个智能助手，能够友好地与用户对话。你生成的对话不能包含emoji。"
        ]

        # 如果存在记忆，将记忆作为动态系统消息插入到对话历史前
        if memory_content:
            # 更强烈的记忆提示，确保 LLM 重点关注
            memory_prompt = f"""
                【重要！背景记忆与上下文】
                以下是从历史对话中提取的关键记忆信息，请务必在回复中体现对用户这些信息：

                {memory_content}

                【关键要求】
                1. 请仔细阅读上述记忆内容
                2. 在回复时适当引用或回应这些记忆中的信息
                3. 让用户感受到你记得他们的事情
                4. 不要完全忽略这些记忆信息
                """
            chat_messages.append(SystemMessage(content=memory_prompt))

        chat_messages.extend(current_messages)
        logger.debug(f"查询内容：{chat_messages}")

        # 调用原始模型生成回复
        response = await self.original_model.ainvoke(chat_messages)

        # 添加AI回复到消息历史
        current_messages.append(response)

        # 兜底机制：如果回复为空或无效，或者工具期望生成回复但模型没生成，则生成默认回复
        if not response.content or not cast(str,response.content).strip():
            logger.warning("Chat 回复为空，生成默认回复")

            # 从消息历史中提取最后调用的工具名称
            last_tool_name = None
            for msg in reversed(current_messages[:-1]):
                if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    last_tool_name = msg.tool_calls[0]["name"]
                    break
                elif msg.type == "tool":
                    # 从 tool message 中提取工具调用信息
                    break

            # 根据工具名称生成更智能的默认回复
            tool_name_to_reply = {
                "tell_joke": "笑话已经讲完了，希望你喜欢！",
                "volume_up": "音量已经调大了。",
                "volume_down": "音量已经调小了。",
                "volume_mute": "已经静音了。",
                "volume_set": "音量已经调整好了。",
                "eye_set": "眼睛亮度已经调整好了。",
                "eye_up": "眼睛亮度已经调亮了。",
                "eye_down": "眼睛亮度已经调暗了。",
                "eye_off": "眼睛已经关闭了。",
                "into_calculator": "计算已完成。",
                "query_time": "时间已经查询到了。",
                "tell_story": "故事已经讲完了。",
                "recite_poetry": "诗已经背完了。",
                "into_baike": "百科知识已经查询到了。",
                "into_idiom": "成语已经查询到了。",
                "exit_voice_message": "已退出留言模式。",
                "into_recall_voice_message": "已切换到撤回留言模式，是否开始撤回留言操作?",
            }

            default_response = (
                tool_name_to_reply.get(
                    last_tool_name,
                    "操作已完成。",
                )
                if last_tool_name
                else "好的，我明白了。"
            )

            # 用默认回复替换空回复
            current_messages[-1] = AIMessage(content=default_response)
            logger.info(f"使用默认回复: {default_response}")

        state["messages"] = current_messages
        return state

    # ── flow_resp ────────────────────────────────────────────────

    async def flow_resp(self, state: BTPState) -> BTPState:
        """No-op node — routing is handled by route_by_phase conditional edge."""
        logger.debug("【NODE】flow_resp")
        return state

    # ── route_by_phase ───────────────────────────────────────────

    async def route_by_phase(self, state: BTPState):
        next_node = state.get("next_action_node")
        if next_node == "input_classifier":
            return "input_classifier"
        if next_node == "negotiation_handler":
            return "negotiation_handler"
        if next_node is None:
            current_messages = list(state["messages"])
            user_message = current_messages[-1].content if current_messages else ""
            reject_prompt = [
                SystemMessage(
                    content=(
                        "判断用户是否明确拒绝继续英语练习。只有当用户明确表示不想玩、不想练、退出等拒绝意图时输出reject，否则输出continue。\n"
                        "只输出一个词：reject 或 continue。"
                    )
                ),
                HumanMessage(content=user_message),
            ]
            try:
                resp = await self.original_model.ainvoke(reject_prompt)
                result = cast(str,resp.content).strip().lower().replace(" ", "").replace("\n", "")
                if result == "reject":
                    return "exit_in_reason"
            except Exception:
                pass
            return "negotiation_handler"
        return "negotiation_handler"

    async def exit_in_reason(self, state: BTPState) -> BTPState:
        """
        非action节点
        根据退出原因，设置退出时的提示信息
        """
        logger.debug("【NODE】exit_in_reason")
        current_messages = list(state["messages"])

        # 使用绑定了工具的模型，让模型自动选择并调用工具
        response = await self.model_with_tools.ainvoke(
            [HumanMessage("退出英语场景模式")]
        )

        # 添加AI响应到消息历史
        current_messages.append(response)
        state["messages"] = current_messages

        # 如果模型调用了工具，执行工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    try:
                        result = await tool.ainvoke(tool_args)
                        state["tool_result"] = str(result)
                        logger.info(f"工具 {tool_name} 执行成功")

                        # 添加工具执行结果到消息历史
                        current_messages.append(
                            ToolMessage(
                                content=result.model_dump_json(),
                                tool_call_id=tool_call["id"],
                            )
                        )
                    except Exception as e:
                        state["tool_result"] = f"工具执行失败: {e!s}"
                        logger.error(f"工具 {tool_name} 执行失败: {e}")
                        current_messages.append(
                            ToolMessage(
                                content=f"工具执行失败: {e!s}",
                                tool_call_id=tool_call["id"],
                            )
                        )
                else:
                    state["tool_result"] = f"未找到工具: {tool_name}"
                    logger.warning(f"未找到工具: {tool_name}")
                    current_messages.append(
                        ToolMessage(
                            content=f"未找到工具: {tool_name}",
                            tool_call_id=tool_call["id"],
                        )
                    )

        state["messages"] = current_messages

        exit_message = "已退出英语场景练习模式。"

        current_messages.append(AIMessage(content=exit_message))
        state["messages"] = current_messages
        return state

    # ── negotiation_handler ──────────────────────────────────────

    async def negotiation_handler(self, state: BTPState) -> BTPState:
        logger.debug("【NODE】negotiation_handler")
        current_messages = list(state["messages"])

        # ── First call: propose a default scene ──────────────────
        if state.get("proposed_scene") is None:
            scene = random.choice(self.scenes)
            state["proposed_scene"] = scene
            self._last_proposed_name = scene["name"]

            scene_desc = await self._generate_scene_desc(scene)
            proposal = (
                f"我们来玩一个英语角色扮演游戏吧！\n"
                f"{scene_desc}\n"
                f"你想选这个场景吗？也可以选别的预设场景，或者自定义一个场景，或者我们自由聊天！"
            )
            current_messages.append(AIMessage(content=proposal))
            state["messages"] = current_messages
            state["next_action_node"] = "negotiation_handler"
            return state

        # ── Subsequent calls: classify user response ──────────────
        user_message = current_messages[-1].content if current_messages else ""
        current_scene = state["proposed_scene"]
        assert current_scene is not None

        # ── Fast path: user is selecting from the scene list ──────
        if state.get("awaiting_scene_selection"):
            state["awaiting_scene_selection"] = False
            scene_list = "\n".join(
                f"{i + 1}. {s['name']}（{s['category']}）"
                for i, s in enumerate(self.scenes)
            )
            match_prompt = [
                SystemMessage(
                    content=(
                        "根据用户的输入，判断用户想选择哪个预设场景。\n"
                        "请输出匹配的场景编号（数字），如果无法匹配则输出0。\n"
                        "只输出一个数字，不要有任何其他文字。\n\n"
                        f"可选场景：\n{scene_list}"
                    )
                ),
                HumanMessage(content=user_message),
            ]
            try:
                resp = await self.original_model.ainvoke(match_prompt)
                idx = int(cast(str,resp.content).strip()) - 1
                matched = self.scenes[idx] if 0 <= idx < len(self.scenes) else None
            except Exception:
                matched = None
            if matched:
                chosen_scene = matched
                state["scene_category"] = chosen_scene["category"]
                state["scene_name"] = chosen_scene["name"]
                state["ai_role"] = chosen_scene["ai_role"]
                state["user_role"] = chosen_scene["user_role"]
                state["mode"] = "scene"
                state["phase"] = "dialogue"
                opening_prompt = [
                    SystemMessage(
                        content=(
                            f"你是{chosen_scene['ai_role']}。根据以下参考开场白，生成一句语义相近但表述不同的英文开场白。\n"
                            "要求：意思一致，但用词和句式要有变化，像自然对话。只输出开场白，不要有其他内容。不要包含表情符号或emoji。"
                        )
                    ),
                    HumanMessage(
                        content=f"参考开场白：{chosen_scene['reference_opening']}"
                    ),
                ]
                try:
                    opening_resp = await self.original_model.ainvoke(opening_prompt)
                    opening_line = cast(str,opening_resp.content).strip().strip('"').strip("'")
                except Exception:
                    opening_line = chosen_scene["reference_opening"]
                if chosen_scene["ai_role"] != chosen_scene["user_role"]:
                    confirm_msg = (
                        f"好，我来当{chosen_scene['ai_role']}，你是{chosen_scene['user_role']}，开始咯！\n"
                        f"{opening_line}"
                    )
                else:
                    confirm_msg = (
                        f"好，我们来扮演{chosen_scene['name']}的场景，开始咯！\n"
                        f"{opening_line}"
                    )
                current_messages.append(AIMessage(content=confirm_msg))
                state["messages"] = current_messages
                state["next_action_node"] = "input_classifier"
                return state
            # No match — generate custom scene from user description
            custom_scene = await self._generate_custom_scene(cast(str,user_message))
            if custom_scene:
                chosen_scene = custom_scene
                state["scene_category"] = chosen_scene["category"]
                state["scene_name"] = chosen_scene["name"]
                state["ai_role"] = chosen_scene["ai_role"]
                state["user_role"] = chosen_scene["user_role"]
                state["mode"] = "scene"
                state["phase"] = "dialogue"
                opening_prompt = [
                    SystemMessage(
                        content=(
                            f"你是{chosen_scene['ai_role']}。根据以下参考开场白，生成一句语义相近但表述不同的英文开场白。\n"
                            "要求：意思一致，但用词和句式要有变化，像自然对话。只输出开场白，不要有其他内容。不要包含表情符号或emoji。"
                        )
                    ),
                    HumanMessage(
                        content=f"参考开场白：{chosen_scene['reference_opening']}"
                    ),
                ]
                try:
                    opening_resp = await self.original_model.ainvoke(opening_prompt)
                    opening_line = cast(str,opening_resp.content).strip().strip('"').strip("'")
                except Exception:
                    opening_line = chosen_scene["reference_opening"]
                if chosen_scene["ai_role"] != chosen_scene["user_role"]:
                    confirm_msg = (
                        f"好，我来当{chosen_scene['ai_role']}，你是{chosen_scene['user_role']}，开始咯！\n"
                        f"{opening_line}"
                    )
                else:
                    confirm_msg = (
                        f"好，我们来扮演{chosen_scene['name']}的场景，开始咯！\n"
                        f"{opening_line}"
                    )
                current_messages.append(AIMessage(content=confirm_msg))
                state["messages"] = current_messages
                state["valid_dialogues"] = list(state.get("valid_dialogues") or []) + [
                    AIMessage(content=confirm_msg)
                ]
                state["next_action_node"] = "input_classifier"
                return state
            # Custom scene generation also failed — fall through to normal classification

        classify_prompt = [
            SystemMessage(
                content=(
                    "你是场景协商阶段的分类器。根据用户输入判断意图。\n\n"
                    "分类标签：\n"
                    "- accept：用户同意默认场景或回复模糊（如'随便''都行''好的''嗯'）\n"
                    "- select_scene：用户指定了一个预设场景名称\n"
                    "- custom_scene：用户描述了一个自定义场景（如'我想练在医院挂号''我要玩在游乐场遇到朋友'）\n"
                    "- custom_request：用户想自定义场景但还没描述具体内容（如'我要自定义一个场景''自己选一个'）\n"
                    "- free_chat：用户想自由聊天，不选场景\n"
                    "- reroll：用户要求换一个场景（如'换一个''别的''再来一个'）\n"
                    "- list_scenes：用户想查看有哪些预设场景可选（如'有什么场景''有哪些''看看场景'）\n"
                    "- non_compliant：请求不合规（涉及恋爱、暴力、色情、危险行为等）\n\n"
                    "请以JSON格式输出分类结果和你的置信度。"
                )
            ),
            HumanMessage(content=user_message),
        ]

        try:
            structured_llm = self.original_model.with_structured_output(
                NegotiationIntent, method="function_calling"
            )
            result = await structured_llm.ainvoke(classify_prompt)
            typed = cast(NegotiationIntent,result)
            classification = typed.intent
            confidence = typed.confidence
            logger.debug(
                f"User message classified as {classification} with confidence {confidence}"
            )
        except Exception:
            classification = "accept"
            confidence = 0.0

        if confidence < 0.6:
            confirm_msg = "你是想选一个场景，还是想自由聊天呢？"
            current_messages.append(AIMessage(content=confirm_msg))
            state["messages"] = current_messages
            state["next_action_node"] = "negotiation_handler"
            return state

        logger.debug(f"User message classified as {classification}")
        # ── Handle each classification ───────────────────────────

        if classification == "non_compliant":
            redirect = "让我们聊点其他的吧！你愿意试试英语角色扮演，还是想自由聊天呢？"
            current_messages.append(AIMessage(content=redirect))
            state["messages"] = current_messages
            state["next_action_node"] = "negotiation_handler"
            return state

        if classification == "reroll":
            available = [
                s for s in self.scenes if s["name"] != self._last_proposed_name
            ]
            if not available:
                available = self.scenes
            scene = random.choice(available)
            state["proposed_scene"] = scene
            self._last_proposed_name = scene["name"]
            reroll_msg = (
                f"好的，换个场景！\n"
                f"{await self._generate_scene_desc(scene)}\n"
                f"想玩这个吗？"
            )
            current_messages.append(AIMessage(content=reroll_msg))
            state["messages"] = current_messages
            state["next_action_node"] = "negotiation_handler"
            return state

        if classification == "list_scenes":
            categories = {}
            for s in self.scenes:
                categories.setdefault(s["category"], []).append(s["name"])
            scene_lines = []
            for cat, names in sorted(categories.items()):
                items = "、".join(names)
                scene_lines.append(f"- {cat}类的有：{items}")
            scene_list = "\n".join(scene_lines)
            list_msg = (
                f"这些是可选的预设场景：\n{scene_list}\n\n"
                f"你想选择哪个场景？也可以自定义一个场景，或者我们自由聊天！"
            )
            current_messages.append(AIMessage(content=list_msg))
            state["messages"] = current_messages
            state["next_action_node"] = "negotiation_handler"
            state["awaiting_scene_selection"] = True
            return state

        if classification == "free_chat":
            state["mode"] = "free_chat"
            state["phase"] = "dialogue"
            current_messages.append(
                AIMessage(
                    content="那，我们自由聊天吧！你可以用英语跟我聊任何你想聊的话题，我会帮你练英语哦！Hello! What would you like to talk about today?"
                )
            )
            state["messages"] = current_messages
            state["next_action_node"] = "input_classifier"
            return state

        # Determine which scene to use
        chosen_scene = current_scene  # default for accept

        if classification == "select_scene":
            scene_list = "\n".join(
                f"{i + 1}. {s['name']}（{s['category']}）"
                for i, s in enumerate(self.scenes)
            )
            match_prompt = [
                SystemMessage(
                    content=(
                        "根据用户的输入，判断用户想选择哪个预设场景。\n"
                        "请输出匹配的场景编号（数字），如果无法匹配则输出0。\n"
                        "只输出一个数字，不要有任何其他文字。\n\n"
                        f"可选场景：\n{scene_list}"
                    )
                ),
                HumanMessage(content=user_message),
            ]
            try:
                logger.debug(f"Match prompt: {match_prompt}")
                resp = await self.original_model.ainvoke(match_prompt)
                logger.debug(f"Match response: {resp}")
                idx = int(cast(str,resp.content).strip()) - 1
                matched = self.scenes[idx] if 0 <= idx < len(self.scenes) else None
            except Exception:
                matched = None
            chosen_scene = matched or current_scene
            if matched is None:
                custom_scene = await self._generate_custom_scene(cast(str,user_message))
                if custom_scene:
                    chosen_scene = custom_scene
                else:
                    current_messages.append(
                        AIMessage(
                            content=f"抱歉，我没找到你说的场景。我推荐当前的【{current_scene['category']}】{current_scene['name']}，可以试试吗？或者换个场景？"
                        )
                    )
                    state["messages"] = current_messages
                    state["next_action_node"] = "negotiation_handler"
                    return state

        if classification == "custom_request":
            current_messages.append(
                AIMessage(
                    content="好的！请告诉我你想玩什么样的场景？比如在哪个地方、做什么事情、和谁一起，随便描述就行。"
                )
            )
            state["messages"] = current_messages
            state["next_action_node"] = "negotiation_handler"
            return state

        if classification == "custom_scene":
            custom_scene = await self._generate_custom_scene(cast(str,user_message))
            chosen_scene = custom_scene if custom_scene else current_scene

        # ── Scene confirmed: set up and generate opening ─────────
        state["scene_category"] = chosen_scene["category"]
        state["scene_name"] = chosen_scene["name"]
        state["ai_role"] = chosen_scene["ai_role"]
        state["user_role"] = chosen_scene["user_role"]
        state["mode"] = "scene"
        state["phase"] = "dialogue"

        # Generate variant opening via LLM
        opening_prompt = [
            SystemMessage(
                content=(
                    f"你是{chosen_scene['ai_role']}。根据以下参考开场白，生成一句语义相近但表述不同的英文开场白。\n"
                    "要求：意思一致，但用词和句式要有变化，像自然对话。只输出开场白，不要有其他内容。不要包含表情符号或emoji。"
                )
            ),
            HumanMessage(content=f"参考开场白：{chosen_scene['reference_opening']}"),
        ]
        try:
            opening_resp = await self.original_model.ainvoke(opening_prompt)
            opening_line = cast(str,opening_resp.content).strip().strip('"').strip("'")
        except Exception:
            opening_line = chosen_scene["reference_opening"]

        if chosen_scene["ai_role"] != chosen_scene["user_role"]:
            confirm_msg = (
                f"好，我来当{chosen_scene['ai_role']}，你是{chosen_scene['user_role']}，开始咯！\n"
                f"{opening_line}"
            )
        else:
            confirm_msg = (
                f"好，我们来扮演{chosen_scene['name']}场景，开始咯！\n{opening_line}"
            )
        current_messages.append(AIMessage(content=confirm_msg))
        state["messages"] = current_messages
        state["valid_dialogues"] = list(state.get("valid_dialogues") or []) + [
            AIMessage(content=confirm_msg)
        ]
        state["next_action_node"] = "input_classifier"
        return state

    # ── input_classifier ─────────────────────────────────────────

    async def input_classifier(self, state: BTPState) -> BTPState:
        logger.debug("【NODE】input_classifier")
        current_messages = list(state["messages"])
        user_message = current_messages[-1].content if current_messages else ""

        # ── Step 1: Compliance check ──────────
        compliance_prompt = [
            SystemMessage(
                content=(
                    "判断用户输入是否不合规（面向5~12岁儿童）。以下任一类则输出unsafe，否则输出safe：\n"
                    "- 恋爱、约会、情侣、亲密关系等情感扮演\n"
                    "- 血腥、暴力、伤害、武器、战争、恐怖、惊悚、灵异、死亡\n"
                    "- 色情、性暗示、身体隐私话题\n"
                    "- 脏话、人身攻击、歧视、霸凌\n"
                    "- 烟、酒、毒品、赌博等成人行为\n"
                    "- 危险行为教唆（玩火、危险动作、自伤等）\n"
                    "- 政治敏感、宗教争议、意识形态对立\n"
                    "- 涉及个人隐私的引导（家庭住址、家长收入、家庭矛盾等）\n"
                    "只输出一个词：unsafe 或 safe。"
                )
            ),
            HumanMessage(content=user_message),
        ]
        try:
            resp = await self.original_model.ainvoke(compliance_prompt)
            safety = cast(str,resp.content).strip().lower()
        except Exception:
            safety = "safe"

        if "unsafe" in safety:
            state["input_type"] = "non_compliant"
            return state

        # ── Step 2: ask_explain 检测 ──────────
        has_chinese = bool(re.search(r"[一-鿿]", cast(str,user_message)))

        if (
            has_chinese
            and len(current_messages) >= 2
            and isinstance(current_messages[-2], AIMessage)
        ):
            prev_ai_msg = current_messages[-2].content
            explain_prompt = [
                SystemMessage(
                    content=(
                        "判断用户是否在询问上一句AI说的英语是什么意思。\n"
                        "是则输出explain，否则输出other。\n"
                        "规则：如果用户在回应、反驳、补充说明自己的想法（即使中英混杂），不算ask_explain。\n"
                        "只有用户明确在追问某个词/句的意思时，才算ask_explain。\n"
                        "示例（explain）：'什么意思' '没听懂' '你说的是什么' '那句话怎么读' '不懂你在说什么'\n"
                        "示例（other）：'什么颜色' '多少钱' '好的' '我想买那个' 'no,my giraffe 是短脖子' '对，但是我不喜欢那个颜色'\n"
                        "只输出一个词。"
                    )
                ),
                HumanMessage(
                    content=f"上一句AI说的是：{prev_ai_msg}\n用户回复：{user_message}"
                ),
            ]
            try:
                resp = await self.original_model.ainvoke(explain_prompt)
                explain = cast(str,resp.content).strip().lower()
            except Exception:
                explain = "other"

            if "explain" in explain:
                state["input_type"] = "ask_explain"
                return state

        # ── Step 3: Input type classification ─────────────────────

        if has_chinese:
            classify_prompt = [
                SystemMessage(
                    content=(
                        "用户输入包含中文。判断属于以下哪种：\n"
                        "- vocab_gap：中英混杂，仅个别词汇缺失，替换后句子通顺。例：'I want to eat 冰淇淋'\n"
                        "- sentence_gap：中文部分较多或替换后不通顺。例：'我想去买一个toy'\n"
                        "- all_chinese：完全是中文。例：'我想吃冰淇淋'\n"
                        "只输出标签，不要有任何其他文字。"
                    )
                ),
                HumanMessage(content=user_message),
            ]
            default_type = "all_chinese"
        else:
            classify_prompt = [
                SystemMessage(
                    content=(
                        "用户输入全是英文。判断属于以下哪种：\n"
                        "- grammar_error：存在语法/时态/单复数/冠词等错误。例：'He go to school yesterday'\n"
                        "注意：大小写（如i写成I）和标点符号不算错误，因为输入来自语音识别，忽略这些。只判断语法层面的错误。\n"
                        "- clean：无错误或仅有极轻微瑕疵\n"
                        "只输出标签，不要有任何其他文字。"
                    )
                ),
                HumanMessage(content=user_message),
            ]
            default_type = "clean"

        try:
            resp = await self.original_model.ainvoke(classify_prompt)
            input_type = cast(str,resp.content).strip().lower().replace(" ", "").replace("\n", "")
        except Exception:
            input_type = default_type

        if has_chinese:
            valid_types = {"vocab_gap", "sentence_gap", "all_chinese"}
        else:
            valid_types = {"grammar_error", "clean"}
        if input_type not in valid_types:
            input_type = default_type

        logger.debug(f"【INPUT】: {input_type}")
        # ── Step 3: Analysis via structured output (skip for clean) ──
        teach_words = None
        correct_sentence = None
        error_explain = None

        if input_type not in ("clean", "ask_explain"):
            task_desc = {
                "vocab_gap": "分析用户的输入，提取缺失的英文词汇并给出正确表达。",
                "sentence_gap": "分析用户的输入，提取关键中文词并给出完整英文表达。",
                "grammar_error": "分析用户的英文输入，给出正确句子和简短解释。注意：大小写和标点符号不算错误，忽略这些，只关注语法层面的错误（时态、单复数、冠词、介词搭配等）。",
                "all_chinese": "用户输入完全是中文。请将用户的中文翻译成简单的英文表达，必须选取1~2个中文词汇给出中英对应。teach_words和correct_sentence都必须填写。",
            }
            system = (
                f"{task_desc.get(input_type, '')}\n"
                "用户是5~12岁的中文母语儿童。以JSON格式输出。\n"
                '- teach_words: 需要教的词汇，格式"中文词"对应的英文是"English word"，多个用逗号分隔。\n'
                "- correct_sentence: 修正后的完整英文句子。无则填null\n"
                "- error_explain: 简短的中文错误解释（给5-12岁孩子看的，不用语法术语）。无则填null"
            )
            try:
                structured_llm = self.original_model_smart.with_structured_output(
                    AnalysisOutput, method="function_calling"
                )
                result = await structured_llm.ainvoke(
                    [
                        SystemMessage(content=system),
                        HumanMessage(content=user_message),
                    ]
                )
                typed = cast(AnalysisOutput,result)
                teach_words = typed.teach_words or None
                correct_sentence = typed.correct_sentence or None
                error_explain = typed.error_explain or None
            except Exception:
                pass

        # ── Step 4: Standardized error key via separate LLM call ──
        error_counts = state.get("error_counts") or {}
        error_key = None
        if input_type in ("vocab_gap", "sentence_gap", "grammar_error", "all_chinese"):
            try:
                structured_llm = self.original_model.with_structured_output(
                    ErrorKeyOutput, method="function_calling"
                )
                result = await structured_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "你是一个错误分类器。根据用户的输入和教学分析结果，输出一个标准化的错误标识符。以JSON格式输出。\n"
                                "规则：\n"
                                "- 格式为 类型:具体内容，全部小写，用下划线连接单词\n"
                                "- vocab 类型：格式 vocab:english_word（取核心英文词，多个词用下划线连接）\n"
                                "- grammar 类型：格式 grammar:具体语法点（如 grammar:past_tense, grammar:plural, grammar:article）\n"
                                "- 同一个语法问题（即使句子不同）应映射到同一个标识符\n"
                                "- 同一个词汇缺失（即使句子不同）应映射到同一个标识符\n"
                                "- 如果有多个错误，选最主要的一个"
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"用户输入: {user_message}\n"
                                f"输入类型: {input_type}\n"
                                f"教词汇: {teach_words or '无'}\n"
                                f"正确句子: {correct_sentence or '无'}\n"
                                f"错误解释: {error_explain or '无'}"
                            )
                        ),
                    ]
                )
                typed = cast(ErrorKeyOutput,result)
                error_key = typed.error_key
            except Exception:
                pass

        # ── Step 5: Error tolerance: same error corrected 3+ times → clean ─
        if error_key and input_type in (
            "vocab_gap",
            "sentence_gap",
            "grammar_error",
            "all_chinese",
        ):
            error_counts = dict(error_counts)
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
            state["error_counts"] = error_counts

            if error_counts[error_key] >= 3:
                input_type = "clean"
                teach_words = None
                correct_sentence = None
                error_explain = None
                logger.info(
                    f"同类错误[{error_key}]已纠正{error_counts[error_key]}次，跳过教学"
                )

        state["input_type"] = input_type
        state["teach_words"] = teach_words
        state["correct_sentence"] = correct_sentence
        state["error_explain"] = error_explain
        return state

    # ── teach_vocab ──────────────────────────────────────────────

    async def teach_vocab(self, state: BTPState) -> BTPState:
        """Strategy A: Only individual words missing."""
        logger.debug("【NODE】teach_vocab")
        current_messages = list(state["messages"])
        user_message = current_messages[-1].content if current_messages else ""

        teach_words = state.get("teach_words") or ""
        correct_sentence = state.get("correct_sentence") or ""

        recent = list(state.get("valid_dialogues") or [])
        context_str = "\n".join(
            f"{'AI' if isinstance(m, AIMessage) else '用户'}: {m.content}"
            for m in recent[-5:]
        )

        prompt = [
            SystemMessage(
                content=(
                    "你是一个温和的英语老师，面对5~12岁的中文母语孩子。孩子说了一句话，其中个别词汇用了中文。\n"
                    "你的任务是：告诉孩子那句中文对应的英文怎么说，然后给出完整的英文表达。\n"
                    "重要：你要教的是中文→英文，不是解释英文单词的中文意思。\n"
                    "规则：\n"
                    "- 不使用语法术语\n"
                    "- 用例子而不是讲解规则\n"
                    "- 最多教1个新词汇\n"
                    "- 控制在2-3句话以内\n"
                    '- 先说"XX"的英文是"YY"，再以"你可以说："为开头说完整句子\n'
                    "- 完整正确表达必须贴合当前对话场景，不要生成无关的例句\n"
                    "- 不要包含任何表情符号或emoji\n\n"
                    f"当前对话上下文：\n{context_str}\n\n"
                    f"孩子说的话：{user_message}\n"
                    f"需要教的词汇：{teach_words}\n"
                    f"完整正确表达：{correct_sentence}"
                )
            ),
        ]
        try:
            resp = await self.original_model_smart.ainvoke(prompt)
            teaching = cast(str,resp.content).strip()
        except Exception:
            teaching = (
                f'你说的"{teach_words}"可以用英语来表达。试试说：{correct_sentence}'
            )

        current_messages.append(AIMessage(content=teaching))
        state["messages"] = current_messages
        state["next_action_node"] = "input_classifier"
        return state

    # ── teach_sentence ───────────────────────────────────────────

    async def teach_sentence(self, state: BTPState) -> BTPState:
        """Strategy B: Sentence structure issues."""
        logger.debug("【NODE】teach_sentence")
        current_messages = list(state["messages"])
        user_message = current_messages[-1].content if current_messages else ""

        teach_words = state.get("teach_words") or ""
        correct_sentence = state.get("correct_sentence") or ""

        recent = list(state.get("valid_dialogues") or [])
        context_str = "\n".join(
            f"{'AI' if isinstance(m, AIMessage) else '用户'}: {m.content}"
            for m in recent[-5:]
        )

        input_type = state.get("input_type", "sentence_gap")
        if input_type == "all_chinese":
            desc = "孩子用纯中文表达了一句话，你需要教孩子这句话用英文怎么说。"
        else:
            desc = "孩子用中英混杂的方式表达了一句话。"

        prompt = [
            SystemMessage(
                content=(
                    f"你是一个温和的英语老师，面对5~12岁的中文母语孩子。{desc}\n"
                    "你的任务是教中文→英文，不是解释英文单词的意思。\n"
                    "全程使用中文进行教学，不要用英文讲解。\n"
                    "规则：\n"
                    "- 不使用语法术语\n"
                    "- 使用简单、常用的词汇\n"
                    "- 控制在2-3句话以内\n"
                    "- 完整英文表达必须贴合当前对话场景，不要生成无关的例句\n"
                    "- 不要包含任何表情符号或emoji\n\n"
                    f"当前对话上下文：\n{context_str}\n\n"
                    f"孩子说的话：{user_message}\n"
                    f"需要教的关键词：{teach_words}\n"
                    f"完整英文表达：{correct_sentence}"
                )
            ),
        ]
        try:
            resp = await self.original_model_smart.ainvoke(prompt)
            teaching = cast(str,resp.content).strip()
        except Exception:
            teaching = f"我们这样说更好：{correct_sentence}"

        current_messages.append(AIMessage(content=teaching))
        state["messages"] = current_messages
        state["next_action_node"] = "input_classifier"
        return state

    # ── teach_grammar ────────────────────────────────────────────

    async def teach_grammar(self, state: BTPState) -> BTPState:
        """Strategy C: Full English but grammar errors."""
        logger.debug("【NODE】teach_grammar")
        current_messages = list(state["messages"])
        user_message = current_messages[-1].content if current_messages else ""

        correct_sentence = state.get("correct_sentence") or ""
        error_explain = state.get("error_explain") or ""

        recent = list(state.get("valid_dialogues") or [])
        context_str = "\n".join(
            f"{'AI' if isinstance(m, AIMessage) else '用户'}: {m.content}"
            for m in recent[-5:]
        )

        prompt = [
            SystemMessage(
                content=(
                    "你是一个温和的英语老师，面对5~12岁的中文母语孩子。孩子用英文表达了一句话，但有语法错误。\n"
                    "你必须先肯定孩子用英文表达的尝试（保护信心），再简短点出问题并给出正确说法。\n"
                    "规则：\n"
                    "- 第1句先肯定孩子的尝试（如'你表达得很棒！'）\n"
                    "- 第2句用中文简短说明（如'注意did后面要用动词原形哦'）\n"
                    '- 第3句以"你可以说："为开头给出完整正确英文句子\n'
                    "- 不使用复杂语法术语\n"
                    "- 大小写和标点符号不算错误，不要纠正这些（输入来自语音识别）\n"
                    "- 控制在2-3句话以内\n"
                    "- 正确表达必须贴合当前对话场景，不要生成无关的例句\n"
                    "- 不要包含任何表情符号或emoji\n\n"
                    f"当前对话上下文：\n{context_str}\n\n"
                    f"孩子说的话：{user_message}\n"
                    f"错误说明：{error_explain}\n"
                    f"正确表达：{correct_sentence}"
                )
            ),
        ]
        try:
            resp = await self.original_model_smart.ainvoke(prompt)
            teaching = cast(str,resp.content).strip()
        except Exception:
            teaching = f"你用英文表达很棒！我们这样说更好：{correct_sentence}"

        current_messages.append(AIMessage(content=teaching))
        state["messages"] = current_messages
        state["next_action_node"] = "input_classifier"
        return state

    # ── compliance_redirect ──────────────────────────────────────

    async def compliance_redirect(self, state: BTPState) -> BTPState:
        logger.debug("【NODE】compliance_redirect")
        current_messages = list(state["messages"])
        phase = state.get("phase", "negotiation")

        if phase == "dialogue":
            ai_role = state.get("ai_role", "小伙伴")
            valid = list(state.get("valid_dialogues") or [])
            redirect_prompt = [
                SystemMessage(
                    content=(
                        f"你是{ai_role}，正在和一个5~12岁的孩子进行英语角色扮演对话。\n"
                        "孩子说了一些不合适的内容，请用英文自然地把话题引到健康的方向。\n"
                        "规则：\n"
                        "- 保持角色身份，不要脱离角色\n"
                        "- 生成的内容中不要包含不合适的词汇\n"
                        "- 不要否定或解释不合适的内容，用1-2句话直接转移话题\n"
                        "- 继续推进对话\n"
                        "- 不要包含任何表情符号或emoji"
                    )
                ),
                *valid[-5:],
                current_messages[-1],
            ]
            next_node = "input_classifier"
            fallback_msg = "That's not something we should talk about. Hey, let me tell you something fun!"
        else:
            redirect_prompt = [
                SystemMessage(
                    content=(
                        "你是一个面向5~12岁儿童的英语陪练伙伴。用户说了以下类型的不合规内容：\n"
                        "恋爱/约会/亲密关系、血腥/暴力/武器、色情/性暗示、恐怖/惊悚/灵异、\n"
                        "脏话/人身攻击/歧视、烟酒毒品/赌博、危险行为教唆、政治敏感/宗教争议、\n"
                        "引导个人隐私（家庭住址、收入等）。\n"
                        "请用结合对话的上下文用，中文温和地转向健康话题，推荐一个替代活动或自由聊天。\n"
                        "规则：\n"
                        "- 不要复述不合适的内容\n"
                        "- 语气友好自然\n"
                        "- 推荐1-2个健康场景或自由聊天\n"
                        "- 控制在1-2句话以内\n"
                        "- 不要包含任何表情符号或emoji"
                    )
                ),
            ]
            next_node = "negotiation_handler"
            fallback_msg = "我们聊点别的吧！你想试试英语角色扮演，还是自由聊天呢？"

        try:
            resp = await self.original_model_smart.ainvoke(redirect_prompt)
            logger.debug(f"【RESPONSE】{resp.content}")
            redirect_msg = cast(str,resp.content).strip()
        except Exception as e:
            logger.debug(f"【ERROR】Failed to generate compliance redirect message.{e}")
            redirect_msg = fallback_msg

        current_messages.append(AIMessage(content=redirect_msg))
        state["messages"] = current_messages
        state["next_action_node"] = next_node
        return state

    # ── dialogue_respond ─────────────────────────────────────────

    async def dialogue_respond(self, state: BTPState) -> BTPState:
        """Unified exit for teaching completion or clean input."""
        logger.debug("【NODE】dialogue_respond")
        current_messages = list(state["messages"])
        user_message = current_messages[-1].content if current_messages else ""

        mode = state.get("mode", "scene")
        ai_role = state.get("ai_role", "小伙伴")

        if mode == "scene":
            system_content = (
                f"你是{ai_role}，正在和一个5~12岁的孩子进行英语角色扮演对话。\n"
                "请用英文推进剧情，自然回应孩子的话。\n"
                "规则：\n"
                "- 保持角色身份\n"
                "- 用简单自然的英语\n"
                "- 回复控制在1-3句话\n"
                "- 不要主动进行教学或纠正\n"
                "- 不要包含任何表情符号或emoji"
            )
        else:
            # Free chat mode
            system_content = (
                "你是一个友好的英语陪练小伙伴，正在和一个5~12岁的孩子自由聊天。\n"
                "请用英文回应孩子的话，跟随话题，适时引入新话题。\n"
                "规则：\n"
                "- 用简单自然的英语\n"
                "- 回复控制在1-3句话\n"
                "- 不要主动进行教学或纠正\n"
                "- 不要包含任何表情符号或emoji"
            )

        # Build prompt with valid dialogue context
        valid = list(state.get("valid_dialogues") or [])
        prompt = [SystemMessage(content=system_content)] + valid[-5:]
        prompt.append(HumanMessage(content=user_message))

        try:
            resp = await self.original_model.ainvoke(prompt)
            reply = cast(str,resp.content).strip()
        except Exception:
            reply = "That's great! Can you tell me more?"

        current_messages.append(AIMessage(content=reply))
        state["messages"] = current_messages
        valid = list(state.get("valid_dialogues") or [])
        valid.append(HumanMessage(content=user_message))
        valid.append(AIMessage(content=reply))
        state["valid_dialogues"] = valid
        state["next_action_node"] = "input_classifier"
        # Clear teaching state for next turn
        state["teach_words"] = None
        state["correct_sentence"] = None
        state["error_explain"] = None
        return state

    # ── explain_meaning ───────────────────────────────────────────

    async def explain_meaning(self, state: BTPState) -> BTPState:
        """用户询问上一句AI英文的含义时，用中文解释。"""
        logger.debug("【NODE】explain_meaning")
        current_messages = list(state["messages"])
        user_message = current_messages[-1].content if current_messages else ""

        # Find the last AIMessage (the English sentence the user is asking about)
        last_ai_msg = None
        for msg in reversed(current_messages[:-1]):
            if isinstance(msg, AIMessage) and cast(str,msg.content).strip():
                last_ai_msg = cast(str,msg.content).strip()
                break

        if not last_ai_msg:
            current_messages.append(
                AIMessage(content="你有什么想问的吗？可以用英语告诉我哦！")
            )
            state["messages"] = current_messages
            state["next_action_node"] = "input_classifier"
            return state

        prompt = [
            SystemMessage(
                content=(
                    "你是一个面向5~12岁儿童的英语陪练伙伴。孩子不理解你刚才说的英文，请用中文解释。\n"
                    "规则：\n"
                    "- 全程使用中文讲解，不要用英文\n"
                    "- 先给出英文句子的中文翻译\n"
                    "- 再用中文简单解释1-2个关键词汇的意思\n"
                    "- 控制在2-3句话以内\n"
                    "- 不要包含任何表情符号或emoji"
                )
            ),
            HumanMessage(
                content=f"你刚才说的英文是：{last_ai_msg}\n孩子问：{user_message}"
            ),
        ]
        logger.debug(f"【explain_meaning PROMPT】{prompt}")
        try:
            resp = await self.original_model.ainvoke(prompt)
            explanation = cast(str,resp.content).strip()
        except Exception:
            explanation = (
                f"我刚才说的是「{last_ai_msg}」，意思是……你可以试着跟着我说一遍哦！"
            )

        current_messages.append(AIMessage(content=explanation))
        state["messages"] = current_messages
        state["next_action_node"] = "input_classifier"
        return state

    # ── route_input_type ──────────────────────────────────────────

    async def route_input_type(self, state: BTPState) -> str:
        input_type = state.get("input_type", "clean")
        if input_type == "non_compliant":
            return "compliance_redirect"
        if input_type == "vocab_gap":
            return "teach_vocab"
        if input_type in ("sentence_gap", "all_chinese"):
            return "teach_sentence"
        if input_type == "grammar_error":
            return "teach_grammar"
        if input_type == "ask_explain":
            return "explain_meaning"
        return "dialogue_respond"


# ── Factory ──────────────────────────────────────────────────────────


async def autonomous(info: dict) -> CompiledStateGraph:
    """创建英语场景练习智能体实例"""
    from flow.common import llm_flash as llm
    from flow.common import llm_flash as llm_smart

    system_prompt = (
        "你是一个面向5~12岁中文母语儿童的英语口语陪练伙伴。"
        "通过模拟真实生活场景或自由对话，帮助孩子在自然对话中提升英语口语能力。"
        "当孩子中英混杂或有语法错误时，温和地进行教学干预。"
    )

    tools = [exit_english_training_partner]
    agent = BTPAutonomous(llm, llm_smart, tools, info, system_prompt)
    return await agent.compile(StateGraph(BTPState), checkpointer=memory) # type: ignore


if __name__ == "__main__":

    async def main():
        # 创建智能体
        info = {
            "gender": "未知",
            "nickn,ame_toy": "matata",
            "nickname_kid": "宝贝",
            "personality": "[]",
            "birthday": "2015-11-02",
            "toy_id": 18,
            "hobbies": "[]",
        }
        thread_id = "1234567890"
        logger.info(f"thread_id: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        messages = []
        agent = await autonomous(info)
        logger.info("启动智能体")
        print("AIMessage: 已切换到英语场景练习模式，是否开始英语练习？")
        i = 0
        human_messages = test_case.get("human_messages", [])
        response = None
        while True:
            if len(human_messages) > i:
                await asyncio.to_thread(print, "用户输入：", human_messages[i])
                user_input = human_messages[i]
            elif len(human_messages) == 0:
                user_input = await asyncio.to_thread(input, "用户输入：")
            else:
                logger.info("测试用例已执行完毕")
                break

            if user_input == "exit":
                break

            logger.info("运行新图")
            messages.append(HumanMessage(content=user_input))
            response = await agent.ainvoke({"messages": messages[-5:]}, config=config) # type: ignore
            logger.info(response)
            messages.append(response["messages"][-1])
            print(f"AIMessage:{response['messages'][-1].content}")

            i += 1

    test_cases = {"人工交互": {}}
    for value in test_cases.values():
        test_case = value

        asyncio.run(main())
        print("--------------------------------------------")
