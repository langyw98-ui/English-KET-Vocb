import asyncio
import random
from typing import Optional

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from flow.agent import memory
from flow.common import logger, llm_flash, llm_plus
from flow.ket_partner.config import load_config
from flow.ket_partner.db import Repos, init_db
from flow.ket_partner.graph import route_after_init, route_by_intent
from flow.ket_partner.input_classifier import IntentClassification, classify_intent
from flow.ket_partner.nodes import apply_mastery_updates, format_output_text
from flow.ket_partner.profile_summarizer import run_profile_summary
from flow.ket_partner.sentence_generator import generate_sentence
from flow.ket_partner.sentence_validator import validate_sentence
from flow.ket_partner.state import BTPKetState
from flow.ket_partner.translation_evaluator import TranslationEval, evaluate_translation
from flow.ket_partner.vocab_selector import select_target_word
from flow.ket_partner.word_meaning_lookup import WordMeaning, lookup_word_meaning


class KETPartnerAgent:
    def __init__(self, llm_flash, llm_smart, repos: Repos, info: dict, config):
        self.llm_flash = llm_flash
        self.llm_smart = llm_smart
        self.repos = repos
        self.info = info
        self.config = config
        self._bg_tasks = set()
        self._recent_scaffolding: list = []

    async def init_state(self, state: BTPKetState) -> dict:
        profile = await self.repos.profile.get()
        await self.repos.profile.update(in_refill_mode=0)
        update = {
            "topic": profile["current_topic"],
            "profile_strategy": profile["dialogue_strategy"],
            "profile_weakness": ",".join(profile["weakness_words"]),
            "last_english_sentence": None,
            "last_target_word": None,
            "last_sentence_words": [],
        }
        if len(state["messages"]) <= 1:
            history = await self.repos.log.recent(limit=5)
            last_ai = await self.repos.log.last_ai_message()
            if last_ai:
                update["last_english_sentence"] = last_ai["content"]
                update["last_sentence_words"] = last_ai["words_used"]
                update["last_target_word"] = last_ai["target_words"][0] if last_ai["target_words"] else None
        return update

    async def classify_intent_node(self, state: BTPKetState) -> dict:
        kid_input = state["messages"][-1].content if state["messages"] else ""
        result = await classify_intent(self.llm_smart, state.get("last_english_sentence"), kid_input)
        return {"intent": result.intent, "asked_word": result.asked_word}

    async def evaluate_translation_node(self, state: BTPKetState) -> dict:
        kid_input = state["messages"][-1].content
        result = await evaluate_translation(
            self.llm_smart,
            sentence=state["last_english_sentence"],
            words=state["last_sentence_words"],
            target=state["last_target_word"],
            kid_input=kid_input,
        )
        return {"wrong_words": result.wrong_words, "correct_meanings": result.correct_meanings}

    async def lookup_target_meaning_node(self, state: BTPKetState) -> dict:
        result = await lookup_word_meaning(
            self.llm_flash, state["last_english_sentence"], state["last_target_word"]
        )
        return {"target_word_meaning": result.meaning}

    async def lookup_asked_meaning_node(self, state: BTPKetState) -> dict:
        result = await lookup_word_meaning(
            self.llm_flash, state["last_english_sentence"], state["asked_word"]
        )
        return {"asked_word_meaning": result.meaning}

    async def update_mastery_node(self, state: BTPKetState) -> dict:
        await apply_mastery_updates(state, self.repos)
        return {}

    async def select_target_word_node(self, state: BTPKetState) -> dict:
        profile = await self.repos.profile.get()
        word = await select_target_word(self.repos, profile, self.config)
        return {"target_word": word}

    async def generate_sentence_node(self, state: BTPKetState) -> dict:
        age = self.info.get("age", 8)
        sentence = await generate_sentence(
            self.llm_flash,
            target=state["target_word"],
            recent_scaffolding=self._recent_scaffolding[-self.config.variety.recent_window:],
            age=age,
            min_words=self.config.sentence.min_words,
            max_words=self.config.sentence.max_words,
        )
        for _ in range(self.config.validate_retry_limit):
            result = await validate_sentence(sentence, self.repos)
            if result.ok:
                break
            sentence = await generate_sentence(
                self.llm_flash,
                target=state["target_word"],
                recent_scaffolding=self._recent_scaffolding[-self.config.variety.recent_window:],
                age=age,
                min_words=self.config.sentence.min_words,
                max_words=self.config.sentence.max_words,
            )
        else:
            logger.warning(f"sentence validation failed after retries; accepting current draft")
        result = await validate_sentence(sentence, self.repos)
        self._recent_scaffolding.extend(result.words_used)
        return {"last_sentence_words": result.words_used, "_pending_sentence": sentence}

    async def format_output_node(self, state: BTPKetState) -> dict:
        sentence = state.get("_pending_sentence") or ""
        text = format_output_text(state, sentence)
        return {
            "messages": [AIMessage(content=text)],
            "_pending_sentence": None,
            "last_english_sentence": sentence,
        }

    async def explain_meaning_node(self, state: BTPKetState) -> dict:
        asked = state["asked_word"]
        meaning = state.get("asked_word_meaning", "")
        text = f'💡 "{asked}" 的意思是「{meaning}」。\n(继续翻译上句吧)'
        return {"messages": [AIMessage(content=text)]}

    async def redirect_to_translate_node(self, state: BTPKetState) -> dict:
        last = state.get("last_english_sentence") or ""
        text = f"我们继续翻译练习吧。\n🔤 上一句:{last}"
        return {"messages": [AIMessage(content=text)]}

    async def compliance_redirect_node(self, state: BTPKetState) -> dict:
        last = state.get("last_english_sentence") or ""
        text = f"我们换个健康的话题继续练习吧。\n🔤 上一句:{last}"
        return {"messages": [AIMessage(content=text)]}

    async def persist_turn_node(self, state: BTPKetState) -> dict:
        user_msg = state["messages"][-2] if len(state["messages"]) >= 2 else None
        ai_msg = state["messages"][-1] if state["messages"] else None
        profile = await self.repos.profile.get()
        turn_id = profile["total_turns"] + 1

        if user_msg and isinstance(user_msg, HumanMessage):
            await self.repos.log.append("user", user_msg.content, words_used=[], turn_id=turn_id)
        if ai_msg and isinstance(ai_msg, AIMessage):
            await self.repos.log.append(
                "ai",
                ai_msg.content,
                words_used=state.get("last_sentence_words") or [],
                target_words=[state["target_word"]] if state.get("target_word") else [],
                turn_id=turn_id,
            )
            for w in (state.get("last_sentence_words") or []):
                await self.repos.stats.increment_exposed(w)

        await self.repos.profile.update(total_turns=turn_id)

        if turn_id % self.config.summary.interval_turns == 0:
            task = asyncio.create_task(self._run_summary_safe())
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        return {}

    async def _run_summary_safe(self) -> None:
        try:
            await run_profile_summary(self.llm_smart, self.repos)
        except Exception as e:
            logger.warning(f"background summary failed: {e}")

    async def compile(self, builder: StateGraph, checkpointer) -> CompiledStateGraph:
        builder.add_node("init_state", self.init_state)
        builder.add_node("classify_intent", self.classify_intent_node)
        builder.add_node("evaluate_translation", self.evaluate_translation_node)
        builder.add_node("lookup_target_meaning", self.lookup_target_meaning_node)
        builder.add_node("lookup_asked_meaning", self.lookup_asked_meaning_node)
        builder.add_node("update_mastery", self.update_mastery_node)
        builder.add_node("select_target_word", self.select_target_word_node)
        builder.add_node("generate_sentence", self.generate_sentence_node)
        builder.add_node("format_output", self.format_output_node)
        builder.add_node("explain_meaning", self.explain_meaning_node)
        builder.add_node("redirect_to_translate", self.redirect_to_translate_node)
        builder.add_node("compliance_redirect", self.compliance_redirect_node)
        builder.add_node("persist_turn", self.persist_turn_node)

        builder.add_conditional_edges(START, route_after_init, {
            "init_state": "init_state",
            "classify_intent": "init_state",
            "select_target_word": "init_state",
        })
        builder.add_edge("init_state", "classify_intent_or_skip")

        # First-turn shortcut
        builder.add_node("classify_intent_or_skip", self._route_after_init_state)
        builder.add_conditional_edges("classify_intent_or_skip", lambda s: route_after_init(s), {
            "classify_intent": "classify_intent",
            "select_target_word": "select_target_word",
        })

        builder.add_conditional_edges("classify_intent", lambda s: self._route_call2(s), {
            "evaluate_translation": "evaluate_translation",
            "lookup_target_meaning": "lookup_target_meaning",
            "lookup_asked_meaning": "lookup_asked_meaning",
            "skip": "update_mastery",
        })

        builder.add_edge("evaluate_translation", "update_mastery")
        builder.add_edge("lookup_target_meaning", "update_mastery")
        builder.add_edge("lookup_asked_meaning", "update_mastery")
        builder.add_edge("update_mastery", "format_output_or_branch")

        builder.add_node("format_output_or_branch", self._passthrough)
        builder.add_conditional_edges("format_output_or_branch", lambda s: route_by_intent(s), {
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

        return builder.compile(checkpointer=checkpointer)

    async def _route_after_init_state(self, state: BTPKetState) -> dict:
        return {}

    async def _passthrough(self, state: BTPKetState) -> dict:
        return {}

    def _route_call2(self, state: BTPKetState) -> str:
        intent = state.get("intent")
        if intent == "translation":
            return "evaluate_translation"
        if intent == "idk":
            return "lookup_target_meaning"
        if intent == "asks_meaning":
            return "lookup_asked_meaning"
        return "skip"


async def build_agent(llm_flash, llm_smart, repos: Repos, info: dict) -> CompiledStateGraph:
    cfg = load_config()
    agent = KETPartnerAgent(llm_flash, llm_smart, repos, info, cfg)
    builder = StateGraph(BTPKetState)
    return await agent.compile(builder, checkpointer=memory)


async def autonomous(info: dict, db_path: str = "ket_partner.db", csv_path: Optional[str] = None) -> CompiledStateGraph:
    repos = await init_db(db_path, csv_path=csv_path)
    return await build_agent(llm_flash, llm_plus, repos, info)
