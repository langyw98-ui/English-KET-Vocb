import asyncio
from typing import Optional

from langchain.messages import AIMessage, HumanMessage
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
from flow.ket_partner.sentence_generator import generate_sentence, rewrite_sentence
from flow.ket_partner.sentence_validator import validate_sentence
from flow.ket_partner.state import BTPKetState
from flow.ket_partner.translation_evaluator import TranslationEval, evaluate_translation
from flow.ket_partner.vocab_selector import select_target_word
from flow.ket_partner.word_meaning_lookup import WordMeaning, lookup_sentence_translation, lookup_word_meaning


class KETPartnerAgent:
    def __init__(self, llm_flash, llm_smart, repos: Repos, info: dict, config):
        self.llm_flash = llm_flash
        self.llm_smart = llm_smart
        self.repos = repos
        self.info = info
        self.config = config
        self._bg_tasks = set()
        # Per-sentence word lists — list[list[str]] so [-recent_window:]
        # gives the last N SENTENCES (flattened for the prompt), not just
        # the last N words of one sentence (the previous bug).
        self._recent_scaffolding: list = []
        # Full sentence strings for hard dedup — passed to the LLM prompt
        # AND checked post-generation to force a regen if the LLM ignores
        # the soft constraint.
        self._recent_sentences: list = []

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
            # Reset each turn — generate_sentence_node sets this True.
            "_exposure_recorded": False,
        }
        # The DB is the single source of truth (spec §11.1). Always re-hydrate
        # from the last AI log row regardless of how many messages were passed
        # in. The previous `len(state["messages"]) <= 1` guard broke under
        # main.py's `messages[-5:]` windowing: on turn 2 the caller passes
        # [Human1, AI1, Human2] (length 3), the guard failed, and the three
        # keys stayed None/[] — causing route_after_init to re-select a target
        # word (infinite first-turn loop) and evaluate_translation to run
        # against an empty sentence (mastery never incremented).
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
        # Filter to last_sentence_words subset. This is the safety net for:
        # 1. Non-KET words — when generate_sentence_node exhausts retries and
        #    accepts a draft with non_ket_words, those words appear in the
        #    displayed sentence. The LLM may flag them. Without this filter
        #    they'd reach apply_delta and create phantom stats rows.
        # 2. LLM using wrong case / inflected forms — canonical lookup maps
        #    "Eat" → "eat" so it matches the canonical form in last_words.
        # 3. LLM inventing words not in the sentence — silently dropped.
        last_words_set = set(state.get("last_sentence_words") or [])
        filtered = []
        for entry in result.wrong_words:
            if entry.word in last_words_set:
                filtered.append(entry)
                continue
            canonical = await self.repos.vocab.get_ket_word(entry.word)
            if canonical and canonical in last_words_set:
                filtered.append(entry.model_copy(update={"word": canonical}))
            else:
                logger.debug(f"evaluate_translation_node: dropping non-KET word '{entry.word}'")
        return {
            "wrong_words": [e.model_dump() for e in filtered],
            "sentence_translation": result.correct_translation,
        }

    async def lookup_target_meaning_node(self, state: BTPKetState) -> dict:
        result = await lookup_sentence_translation(
            self.llm_flash, state["last_english_sentence"]
        )
        return {"sentence_translation": result.translation}

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
        window = self.config.variety.recent_window
        # Flatten last N sentences' worth of words (was previously a flat
        # list sliced to its tail — only got the last 3 WORDS, not last 3
        # sentences). Now: list[list[str]] outer, flattened here.
        avoid_words = [w for sent_words in self._recent_scaffolding[-window:] for w in sent_words]
        avoid_sentences = list(self._recent_sentences[-window:])

        sentence = await generate_sentence(
            self.llm_flash,
            target=state["target_word"],
            recent_scaffolding=avoid_words,
            age=age,
            min_words=self.config.sentence.min_words,
            max_words=self.config.sentence.max_words,
            avoid_sentences=avoid_sentences,
        )
        result = None
        for _ in range(self.config.validate_retry_limit):
            result = await validate_sentence(sentence, self.repos)
            is_duplicate = sentence in self._recent_sentences
            logger.debug(f"validate_sentence: {result} duplicate={is_duplicate}")
            if result.ok and not is_duplicate:
                break
            # Choose retry path:
            #   - If the sentence passed KET validation but is a duplicate,
            #     force a FULL regen (rewrite can't reliably escape an exact
            #     match — the LLM tends to keep the same scaffold).
            #   - Else if few non-KET words, targeted rewrite.
            #   - Else (many non-KET) full regen.
            if is_duplicate:
                logger.debug(f"regen: sentence is a duplicate of a recent one")
                sentence = await generate_sentence(
                    self.llm_flash,
                    target=state["target_word"],
                    recent_scaffolding=avoid_words,
                    age=age,
                    min_words=self.config.sentence.min_words,
                    max_words=self.config.sentence.max_words,
                    avoid_sentences=avoid_sentences,
                )
            elif result.non_ket_words and len(result.non_ket_words) <= self.config.sentence.rewrite_threshold:
                logger.debug(f"rewrite_sentence: replacing {result.non_ket_words}")
                sentence = await rewrite_sentence(
                    self.llm_flash,
                    original=sentence,
                    replace_words=result.non_ket_words,
                    target=state["target_word"],
                    age=age,
                    min_words=self.config.sentence.min_words,
                    max_words=self.config.sentence.max_words,
                    avoid_sentences=avoid_sentences,
                )
            else:
                sentence = await generate_sentence(
                    self.llm_flash,
                    target=state["target_word"],
                    recent_scaffolding=avoid_words,
                    age=age,
                    min_words=self.config.sentence.min_words,
                    max_words=self.config.sentence.max_words,
                    avoid_sentences=avoid_sentences,
                )
        else:
            logger.warning(f"sentence validation failed after retries; accepting current draft")
            result = await validate_sentence(sentence, self.repos)
            logger.debug(f"validate_sentence: {result} duplicate={is_duplicate}")
        self._recent_scaffolding.append(result.words_used)
        self._recent_sentences.append(sentence)
        # Per spec §11.9, exposed_count is incremented once per word in the
        # NEW sentence. Do this here (on the generate path) so non-generate
        # turns do not re-count the prior sentence's words. Set the flag so
        # persist_turn_node knows the increment is already done and skips its
        # own increment.
        for w in result.words_used:
            await self.repos.stats.increment_exposed(w)
        # Fold the pending sentence directly into last_english_sentence
        # (declared in BTPKetState). The previous `_pending_sentence` hand-off
        # was undeclared and silently dropped by LangGraph, so the kid was
        # shown an empty sentence to translate.
        return {
            "last_sentence_words": result.words_used,
            "last_english_sentence": sentence,
            "_exposure_recorded": True,
        }

    async def format_output_node(self, state: BTPKetState) -> dict:
        # Read the sentence from last_english_sentence (set by
        # generate_sentence_node). For non-generate branches the value is
        # whatever was re-hydrated by init_state from the DB log.
        sentence = state.get("last_english_sentence") or ""
        text = format_output_text(state, sentence)
        # SPREAD existing messages into the return so the REPLACE reducer on
        # `messages` (no reducer annotation in BTPKetState) preserves the
        # caller's HumanMessage(s). Otherwise persist_turn_node only sees the
        # single AIMessage and messages[-2] is out of range — the user input
        # for this turn is then never written to conversation_log.
        return {
            "messages": [*state["messages"], AIMessage(content=text)],
        }

    async def explain_meaning_node(self, state: BTPKetState) -> dict:
        asked = state["asked_word"]
        meaning = state.get("asked_word_meaning", "")
        text = f'"{asked}" 的意思是「{meaning}」。\n(继续翻译上句吧)'
        return {"messages": [*state["messages"], AIMessage(content=text)]}

    async def redirect_to_translate_node(self, state: BTPKetState) -> dict:
        last = state.get("last_english_sentence") or ""
        text = f"我们继续翻译练习吧。\n上一句:{last}"
        return {"messages": [*state["messages"], AIMessage(content=text)]}

    async def compliance_redirect_node(self, state: BTPKetState) -> dict:
        last = state.get("last_english_sentence") or ""
        text = f"我们换个健康的话题继续练习吧。\n上一句:{last}"
        return {"messages": [*state["messages"], AIMessage(content=text)]}

    async def persist_turn_node(self, state: BTPKetState) -> dict:
        user_msg = state["messages"][-2] if len(state["messages"]) >= 2 else None
        ai_msg = state["messages"][-1] if state["messages"] else None
        profile = await self.repos.profile.get()
        turn_id = profile["total_turns"] + 1

        if user_msg and isinstance(user_msg, HumanMessage):
            await self.repos.log.append("user", user_msg.content, words_used=[], turn_id=turn_id)
        if ai_msg and isinstance(ai_msg, AIMessage):
            # Only the generate path produces a sentence whose words count as
            # "actual_words_used" (spec §11.9). When _exposure_recorded is True,
            # generate_sentence_node has ALREADY incremented exposed_count and
            # we log the words on the AI row. For all other branches
            # (asks_meaning / idk / off_topic / non_compliant / redirect) the
            # AI message is not a fresh sentence, so words_used=[] and no
            # increment happens here.
            exposure_recorded = bool(state.get("_exposure_recorded"))
            words_for_log = state.get("last_sentence_words") or [] if exposure_recorded else []
            await self.repos.log.append(
                "ai",
                ai_msg.content,
                words_used=words_for_log,
                target_words=[state["target_word"]] if state.get("target_word") else [],
                turn_id=turn_id,
            )
            # Increment has already happened in generate_sentence_node; do
            # NOT re-increment here.

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

    async def aclose(self, timeout: float = 2.0) -> None:
        """Drain in-flight background summary tasks.

        Without this, a summary task scheduled on the turn that triggers
        /exit would be silently dropped when the event loop closes. We give
        it a small window (default 2s) and swallow exceptions (the summary
        is best-effort by design — see _run_summary_safe).
        """
        if not self._bg_tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._bg_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"aclose: {len(self._bg_tasks)} background task(s) did not finish within {timeout}s; cancelling"
            )
            for t in self._bg_tasks:
                t.cancel()

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
    """Build the compiled graph.

    The underlying KETPartnerAgent is attached to the returned graph object
    as `.agent` so callers (e.g. main.py) can drain background tasks on
    shutdown via `await graph.agent.aclose()`. Attaching it as an attribute
    avoids changing the function's return signature, which would break the
    10+ test callsites that do `agent = await build_agent(...)` then
    `agent.ainvoke(...)`.
    """
    cfg = load_config()
    agent = KETPartnerAgent(llm_flash, llm_smart, repos, info, cfg)
    builder = StateGraph(BTPKetState)
    graph = await agent.compile(builder, checkpointer=memory)
    # Attach the agent instance so main.py's finally block can drain
    # background tasks. (CompiledStateGraph is a Runnable; extra attributes
    # are tolerated by LangGraph.)
    graph.agent = agent  # type: ignore[attr-defined]
    return graph


async def autonomous(info: dict, db_path: str = "ket_partner.db", csv_path: Optional[str] = None) -> CompiledStateGraph:
    """Build the agent for the REPL entrypoint.

    The returned graph has `.agent` attached — main.py uses it to call
    `aclose()` on shutdown.
    """
    repos = await init_db(db_path, csv_path=csv_path)
    # Mark session boundary so last_ai_message() ignores rows from prior
    # sessions. Without this, a kid who exits mid-sentence sees the
    # explanation of that unfinished sentence on restart (init_state
    # would restore it, classify_intent would default to translation/idk,
    # and the answer would leak).
    await repos.log.append_session_start()
    return await build_agent(llm_flash, llm_plus, repos, info)
