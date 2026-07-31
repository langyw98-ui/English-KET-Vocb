"""Graph topology + routing + factory. Extracted from KETPartnerAgent.compile."""
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from flow.common import LlmService, default_llm_service
from flow.ket_partner.agent import KETPartnerAgent
from flow.ket_partner.config import load_config
from flow.ket_partner.state import (
    ASKS_MEANING,
    IDK,
    NON_COMPLIANT,
    OFF_TOPIC,
    TRANSLATION,
    BTPKetState,
    KetIntent,
)

# classify_intent_node 之后路由表:每个 intent 对应下一个节点名。
# OFF_TOPIC / NON_COMPLIANT 不在此分支,route_after_classify 返回 _DEFAULT_AFTER_CLASSIFY。
_ROUTE_AFTER_CLASSIFY: dict[KetIntent, str] = {
    TRANSLATION: "evaluate_translation",
    IDK: "lookup_target_meaning",
    ASKS_MEANING: "lookup_asked_meaning",
}
_DEFAULT_AFTER_CLASSIFY = "skip"


def route_by_intent(state: BTPKetState) -> str:
    """format_output_or_branch 之后路由,根据 intent 选下一节点。"""
    intent = state.get("intent")
    if intent in (TRANSLATION, IDK):
        return "select_target_word"
    if intent == ASKS_MEANING:
        return "explain_meaning"
    if intent == OFF_TOPIC:
        return "redirect_to_translate"
    if intent == NON_COMPLIANT:
        return "compliance_redirect"
    return "select_target_word"


def route_after_init(state: BTPKetState) -> str:
    if state.get("last_english_sentence") is None:
        return "select_target_word"
    return "classify_intent"


def route_after_classify(state: BTPKetState) -> str:
    intent = state.get("intent")
    if intent is None:
        return _DEFAULT_AFTER_CLASSIFY
    return _ROUTE_AFTER_CLASSIFY.get(intent, _DEFAULT_AFTER_CLASSIFY)


async def passthrough_node(state: BTPKetState, config: RunnableConfig) -> dict:
    return {}


def wire_graph(builder: StateGraph, agent: KETPartnerAgent) -> None:
    builder.add_node("init_state", agent.init_state)
    builder.add_node("classify_intent", agent.classify_intent_node)
    builder.add_node("evaluate_translation", agent.evaluate_translation_node)
    builder.add_node("lookup_target_meaning", agent.lookup_target_meaning_node)
    builder.add_node("lookup_asked_meaning", agent.lookup_asked_meaning_node)
    builder.add_node("update_mastery", agent.update_mastery_node)
    builder.add_node("select_target_word", agent.select_target_word_node)
    builder.add_node("generate_sentence", agent.generate_sentence_node)
    builder.add_node("format_output", agent.format_output_node)
    builder.add_node("explain_meaning", agent.explain_meaning_node)
    builder.add_node("redirect_to_translate", agent.redirect_to_translate_node)
    builder.add_node("compliance_redirect", agent.compliance_redirect_node)
    builder.add_node("persist_turn", agent.persist_turn_node)

    builder.add_conditional_edges(START, route_after_init, {
        "init_state": "init_state",
        "classify_intent": "init_state",
        "select_target_word": "init_state",
    })
    builder.add_node("classify_intent_or_skip", passthrough_node)
    builder.add_edge("init_state", "classify_intent_or_skip")
    builder.add_conditional_edges("classify_intent_or_skip", route_after_init, {
        "classify_intent": "classify_intent",
        "select_target_word": "select_target_word",
    })
    builder.add_conditional_edges("classify_intent", route_after_classify, {
        "evaluate_translation": "evaluate_translation",
        "lookup_target_meaning": "lookup_target_meaning",
        "lookup_asked_meaning": "lookup_asked_meaning",
        "skip": "update_mastery",
    })
    builder.add_edge("evaluate_translation", "update_mastery")
    builder.add_edge("lookup_target_meaning", "update_mastery")
    builder.add_edge("lookup_asked_meaning", "update_mastery")
    builder.add_edge("update_mastery", "format_output_or_branch")
    builder.add_node("format_output_or_branch", passthrough_node)
    builder.add_conditional_edges("format_output_or_branch", route_by_intent, {
        "select_target_word": "select_target_word",
        "explain_meaning": "explain_meaning",
        "redirect_to_translate": "redirect_to_translate",
        "compliance_redirect": "compliance_redirect",
    })
    builder.add_edge("select_target_word", "generate_sentence")
    builder.add_edge("generate_sentence", "format_output")
    builder.add_edge("format_output", "persist_turn")
    builder.add_edge("persist_turn", END)
    builder.add_edge("explain_meaning", "persist_turn")
    builder.add_edge("redirect_to_translate", "persist_turn")
    builder.add_edge("compliance_redirect", "persist_turn")


async def build_agent(
    llm_service: LlmService | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    if llm_service is None:
        llm_service = default_llm_service
    cfg = load_config()
    agent = KETPartnerAgent(llm_service, cfg)
    builder = StateGraph(BTPKetState)
    wire_graph(builder, agent)
    graph = builder.compile(checkpointer=checkpointer)
    # CompiledStateGraph 没有声明 .agent 属性;用 setattr 显式挂载 inner agent
    # 实例,供 api/app.py 与 cli/main.py 在 shutdown 时 await .agent.aclose()。
    # 不用 # type: ignore[attr-defined],符合 CLAUDE.md §九.4(仅 Wrapper 模块允许)。
    setattr(graph, "agent", agent)
    return graph
