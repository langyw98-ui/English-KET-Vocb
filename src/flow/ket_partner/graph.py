"""Graph topology + routing + factory. Extracted from KETPartnerAgent.compile."""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from flow.ket_partner.agent import KETPartnerAgent
from flow.ket_partner.config import load_config
from flow.ket_partner.state import BTPKetState


def route_by_intent(state: BTPKetState) -> str:
    intent = state.get("intent")
    if intent in ("translation", "idk"):
        return "select_target_word"
    if intent == "asks_meaning":
        return "explain_meaning"
    if intent == "off_topic":
        return "redirect_to_translate"
    if intent == "non_compliant":
        return "compliance_redirect"
    return "select_target_word"


def route_after_init(state: BTPKetState) -> str:
    if state.get("last_english_sentence") is None:
        return "select_target_word"
    return "classify_intent"


def route_after_classify(state: BTPKetState) -> str:
    """Renamed from KETPartnerAgent._route_call2."""
    intent = state.get("intent")
    if intent == "translation":
        return "evaluate_translation"
    if intent == "idk":
        return "lookup_target_meaning"
    if intent == "asks_meaning":
        return "lookup_asked_meaning"
    return "skip"


async def passthrough_node(state: BTPKetState, config: RunnableConfig) -> dict:
    """No-op node for conditional_edges branching host. Merges _passthrough
    + _route_after_init_state (both were no-ops)."""
    return {}


def wire_graph(builder: StateGraph, agent: KETPartnerAgent) -> None:
    """Add all 13 nodes + edges. Extracted from KETPartnerAgent.compile body."""
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
    llm_flash: BaseChatModel,
    llm_smart: BaseChatModel,
    db: Any = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Factory: load_config -> KETPartnerAgent -> StateGraph -> wire_graph -> compile.
    Attaches .agent to the compiled graph for shutdown lifecycle.

    The ``db`` argument is accepted for backward compatibility with existing
    callers (main.py, app.py, tests) but is not used here: repos are sourced
    per-call from ``config["configurable"]["repos"]`` inside node methods.
    """
    cfg = load_config()
    agent = KETPartnerAgent(llm_flash, llm_smart, cfg)
    builder = StateGraph(BTPKetState)
    wire_graph(builder, agent)
    graph = builder.compile(checkpointer=checkpointer)
    graph.agent = agent  # type: ignore[attr-defined]
    return graph
