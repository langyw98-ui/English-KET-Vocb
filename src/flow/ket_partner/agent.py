import asyncio

from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from flow.common import llm_flash, llm_max, logger
from flow.ket_partner.config import load_config
from flow.ket_partner.db import Repos, init_db
from flow.ket_partner.graph import route_after_init, route_by_intent
from flow.ket_partner.input_classifier import classify_intent
from flow.ket_partner.multi_word_target import target_in_sentence
from flow.ket_partner.nodes import apply_mastery_updates, format_output_text
from flow.ket_partner.profile_summarizer import run_profile_summary
from flow.ket_partner.sentence_generator import generate_sentence
from flow.ket_partner.sentence_naturalness import check_naturalness
from flow.ket_partner.sentence_validator import _tokenize, validate_sentence
from flow.ket_partner.state import BTPKetState
from flow.ket_partner.translation_evaluator import evaluate_translation
from flow.ket_partner.vocab_selector import select_target_word
from flow.ket_partner.word_meaning_lookup import (
    lookup_sentence_translation,
    lookup_word_meaning,
    lookup_word_meanings,
)


class KETPartnerAgent:
    def __init__(self, llm_flash, llm_smart, config):
        self.llm_flash = llm_flash
        self.llm_smart = llm_smart
        self.config = config
        self._bg_tasks = set()

    async def init_state(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos: Repos = config["configurable"]["repos"]
        profile = await repos.profile.get()
        await repos.profile.update(in_refill_mode=0)
        update = {
            "topic": profile["current_topic"],
            "profile_strategy": profile["dialogue_strategy"],
            "profile_weakness": ",".join(profile["weakness_words"]),
            "last_english_sentence": None,
            "last_target_word": None,
            "last_sentence_words": [],
            "_exposure_recorded": False,
        }
        if len(state.get("messages", [])) > 10:
            update["messages"] = state["messages"][-10:]

        last_ai = await repos.log.last_ai_message()
        if last_ai:
            update["last_english_sentence"] = last_ai["content"]
            update["last_sentence_words"] = last_ai["words_used"]
            if last_ai["target_words"]:
                update["last_target_word"] = last_ai["target_words"][0].get("word")
                update["last_target_context"] = last_ai["target_words"][0].get("context", "")
            else:
                update["last_target_word"] = None
                update["last_target_context"] = None
        else:
            update["last_target_context"] = None
        return update

    async def classify_intent_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        kid_input = state["messages"][-1].content if state["messages"] else ""
        result = await classify_intent(self.llm_smart, state.get("last_english_sentence"), kid_input)
        return {"intent": result.intent, "asked_word": result.asked_word}

    async def evaluate_translation_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos: Repos = config["configurable"]["repos"]
        kid_input = state["messages"][-1].content
        result = await evaluate_translation(
            self.llm_smart,
            sentence=state["last_english_sentence"],
            words=state["last_sentence_words"],
            target=state["last_target_word"],
            target_context=state.get("last_target_context") or "",
            kid_input=kid_input,
        )
        logger.debug(f"evaluate_translation_node: {result}")
        last_words_set = set(state.get("last_sentence_words") or [])
        last_sentence = state.get("last_english_sentence") or ""
        displayed_tokens = {
            t.lower().rstrip(".?!") for t in _tokenize(last_sentence)
        } if last_sentence else set()
        seen_words: set[str] = set()
        filtered = []
        for entry in result.wrong_words:
            if (
                entry.kid_translation
                and entry.correct_translation
                and entry.kid_translation == entry.correct_translation
            ):
                logger.debug(
                    f"evaluate_translation_node: dropping '{entry.word}' — "
                    f"kid_translation matches correct_translation ({entry.kid_translation!r})"
                )
                continue
            if not entry.correct_translation:
                logger.debug(
                    f"evaluate_translation_node: dropping '{entry.word}' — "
                    f"empty correct_translation (likely a function word with no single Chinese equivalent)"
                )
                continue
            if entry.word not in last_words_set:
                wr = await repos.vocab.get_ket_word_any_context(entry.word)
                if wr and wr.word in last_words_set:
                    entry = entry.model_copy(update={"word": wr.word})
            if entry.word in last_words_set or entry.word.lower() in displayed_tokens:
                key = entry.word
            else:
                logger.debug(
                    f"evaluate_translation_node: dropping '{entry.word}' — "
                    f"not in displayed sentence"
                )
                continue
            if key in seen_words:
                logger.debug(f"evaluate_translation_node: dropping duplicate wrong_word '{key}'")
                continue
            seen_words.add(key)
            filtered.append(entry)
        return {
            "wrong_words": [e.model_dump() for e in filtered],
            "sentence_translation": result.correct_translation,
            "overall_correct": result.overall_correct,
        }

    async def lookup_target_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        result = await lookup_sentence_translation(
            self.llm_flash, state["last_english_sentence"]
        )
        return {"sentence_translation": result.translation}

    async def lookup_asked_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        result = await lookup_word_meaning(
            self.llm_flash, state["last_english_sentence"], state["asked_word"]
        )
        return {"asked_word_meaning": result.meaning}

    async def update_mastery_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos: Repos = config["configurable"]["repos"]
        await apply_mastery_updates(state, repos)
        return {}

    async def select_target_word_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos: Repos = config["configurable"]["repos"]
        profile = await repos.profile.get()
        word_ref = await select_target_word(repos, profile, self.config)
        if word_ref is None:
            return {"target_word": None, "target_context": None}
        return {"target_word": word_ref.word, "target_context": word_ref.context}

    async def generate_sentence_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos: Repos = config["configurable"]["repos"]
        user_info: dict = config["configurable"].get("user_info", {})
        age = user_info.get("age", 8)
        window = self.config.variety.recent_window
        avoid_words = [w for sent_words in (await repos.recent.list_recent_scaffolding(window=window)) for w in sent_words]
        avoid_sentences = await repos.recent.list_recent(limit=window)
        profile = await repos.profile.get()

        target = state["target_word"]
        target_ctx = state.get("target_context") or ""

        sentence, result, final_target, final_ctx = await self._generate_with_fallback(
            state["target_word"], target_ctx, avoid_words, avoid_sentences, age, profile, repos
        )
        target, target_ctx = final_target, final_ctx

        await repos.recent.append(sentence, window=window)

        if (
            target
            and target not in result.words_used
            and target_in_sentence(target, sentence)
        ):
            result.words_used.append(target)
            constituents = {c.lower() for c in target.split()}
            result.words_used = [
                w for w in result.words_used
                if w == target or w.lower() not in constituents
            ]
            result.non_ket_words = [
                w for w in result.non_ket_words
                if w.lower() not in constituents
            ]
            logger.debug(
                f"multi-word target patch: added '{target}', "
                f"final words_used={result.words_used}, "
                f"non_ket_words={result.non_ket_words}"
            )
        for w in result.words_used:
            ctx = target_ctx if w == target else ""
            await repos.stats.increment_exposed(w, context=ctx, is_target=(w == target))
        annotations: list[dict[str, str]] = []
        if result.non_ket_words:
            annotations = await lookup_word_meanings(
                self.llm_flash, sentence, result.non_ket_words
            )
        update = {
            "last_sentence_words": result.words_used,
            "last_english_sentence": sentence,
            "_exposure_recorded": True,
            "non_ket_annotations": annotations,
        }
        if target != state["target_word"] or target_ctx != (state.get("target_context") or ""):
            update["target_word"] = target
            update["target_context"] = target_ctx
        return update

    async def _validate_and_categorize(
        self, sentence: str, target: str, age: int, repos: Repos, avoid_sentences: list
    ) -> dict:
        result = await validate_sentence(sentence, repos, target=target)
        is_duplicate = sentence in avoid_sentences
        non_ket_count = len(result.non_ket_words)
        is_target_split = (
            bool(target)
            and " " in target.strip()
            and not target_in_sentence(target, sentence)
        )

        passed = False
        reason_kind = None
        reason_detail = ""

        if non_ket_count <= 1 and not is_duplicate and not is_target_split:
            if non_ket_count == 0:
                naturalness = await check_naturalness(self.llm_smart, sentence, age=age)
                logger.debug(
                    f"validate_sentence: {result} duplicate={is_duplicate} "
                    f"target_split={is_target_split} naturalness_ok={naturalness.ok}"
                )
                if naturalness.ok:
                    passed = True
                else:
                    reason_kind = "naturalness"
                    reason_detail = f"unnatural expression — {naturalness.reason}"
            else:
                logger.debug(f"{result} accept: 1 non-KET word will annotate")
                passed = True
        else:
            logger.debug(
                f"validate_sentence: {result} duplicate={is_duplicate} target_split={is_target_split}"
            )
            if is_target_split:
                reason_kind = "target_split"
                reason_detail = f"split the multi-word target '{target}' — words must be contiguous"
            elif is_duplicate:
                reason_kind = "duplicate"
                reason_detail = "word-for-word duplicate of a recent sentence"
            else:
                reason_kind = "non_ket_overflow"
                reason_detail = f"non-KET words {result.non_ket_words} exceed the limit (max 1 allowed)"

        return {
            "result": result,
            "passed": passed,
            "reason_kind": reason_kind,
            "reason_detail": reason_detail,
            "non_ket_words": list(result.non_ket_words),
            "non_ket_count": non_ket_count,
            "is_duplicate": is_duplicate,
            "is_target_split": is_target_split,
            "sentence": sentence,
        }

    async def _generate_with_fallback(
        self,
        initial_target: str,
        initial_context: str,
        avoid_words: list,
        avoid_sentences: list,
        age: int,
        profile: dict,
        repos: Repos,
    ):
        target = initial_target
        context = initial_context
        word_switched = False

        while True:
            attempts: list[dict] = []
            seen_non_ket_words: list = []

            def _regen():
                return generate_sentence(
                    self.llm_smart,
                    target=target,
                    recent_scaffolding=avoid_words,
                    age=age,
                    min_words=self.config.sentence.min_words,
                    max_words=self.config.sentence.max_words,
                    avoid_sentences=avoid_sentences,
                    prior_attempts=attempts,
                    avoid_non_ket_words=seen_non_ket_words,
                    target_context=context,
                )

            sentence = await _regen()
            result = None

            for _ in range(self.config.validate_retry_limit):
                check = await self._validate_and_categorize(sentence, target, age, repos, avoid_sentences)
                result = check["result"]
                if check["passed"]:
                    return sentence, result, target, context
                attempts.append({
                    "sentence": sentence,
                    "reason_kind": check["reason_kind"],
                    "reason_detail": check["reason_detail"],
                    "non_ket_words": check["non_ket_words"],
                    "non_ket_count": check["non_ket_count"],
                })
                for w in check["non_ket_words"]:
                    if w not in seen_non_ket_words:
                        seen_non_ket_words.append(w)
                sentence = await _regen()

            check = await self._validate_and_categorize(sentence, target, age, repos, avoid_sentences)
            result = check["result"]
            if check["passed"]:
                return sentence, result, target, context
            attempts.append({
                "sentence": sentence,
                "reason_kind": check["reason_kind"],
                "reason_detail": check["reason_detail"],
                "non_ket_words": check["non_ket_words"],
                "non_ket_count": check["non_ket_count"],
            })

            overflow_attempts = [a for a in attempts if a["reason_kind"] == "non_ket_overflow"]
            all_naturalness = bool(attempts) and all(
                a["reason_kind"] == "naturalness" for a in attempts
            )

            if overflow_attempts:
                best = min(reversed(overflow_attempts), key=lambda a: a["non_ket_count"])
                sentence = best["sentence"]
                result = await validate_sentence(sentence, repos, target=target)
                logger.warning(
                    f"sentence validation: accepting non-KET overflow draft after "
                    f"{len(attempts)} attempts (non_ket_count={len(result.non_ket_words)}); "
                    f"sentence={sentence!r}"
                )
                return sentence, result, target, context
            elif all_naturalness and not word_switched:
                logger.info(
                    f"all {len(attempts)} attempts failed on naturalness; "
                    f"switching target word from '{target}'"
                )
                new_ref = await select_target_word(repos, profile, self.config)
                if new_ref is None or new_ref.word == target:
                    logger.warning(
                        f"could not find a different target word; "
                        f"accepting final draft: {sentence!r}"
                    )
                    return sentence, result, target, context
                target = new_ref.word
                context = new_ref.context
                word_switched = True
                continue
            else:
                reasons = []
                if check["non_ket_count"] > 1:
                    reasons.append(f"{check['non_ket_count']} non-KET word(s): {check['non_ket_words']}")
                if check["is_duplicate"]:
                    reasons.append("duplicate of a recent sentence")
                if check["is_target_split"]:
                    reasons.append(f"multi-word target '{target}' split apart")
                if not reasons:
                    reasons.append(f"naturalness: {check['reason_detail']}")
                logger.warning(
                    f"sentence validation failed after {len(attempts)} attempts; "
                    f"accepting current draft — reasons: {('; '.join(reasons)) or 'unknown'}; "
                    f"sentence={sentence!r}"
                )
                return sentence, result, target, context

    async def format_output_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        sentence = state.get("last_english_sentence") or ""
        text = format_output_text(state, sentence)
        return {
            "messages": [*state["messages"], AIMessage(content=text)],
        }

    async def explain_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        asked = state["asked_word"]
        last = state.get("last_english_sentence") or ""
        meaning = state.get("asked_word_meaning", "")
        text = f'"{asked}" 的意思是「{meaning}」。\n让我们继续吧，{last}。'
        return {"messages": [*state["messages"], AIMessage(content=text)]}

    async def redirect_to_translate_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        last = state.get("last_english_sentence") or ""
        text = f"我们继续翻译练习吧。\n上一句:{last}"
        return {"messages": [*state["messages"], AIMessage(content=text)]}

    async def compliance_redirect_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        last = state.get("last_english_sentence") or ""
        text = f"我们换个健康的话题继续练习吧。\n上一句:{last}"
        return {"messages": [*state["messages"], AIMessage(content=text)]}

    async def persist_turn_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos: Repos = config["configurable"]["repos"]
        user_msg = state["messages"][-2] if len(state["messages"]) >= 2 else None
        ai_msg = state["messages"][-1] if state["messages"] else None
        profile = await repos.profile.get()
        turn_id = profile["total_turns"] + 1

        if user_msg and isinstance(user_msg, HumanMessage):
            await repos.log.append("user", user_msg.content, words_used=[], turn_id=turn_id)
        if ai_msg and isinstance(ai_msg, AIMessage):
            exposure_recorded = bool(state.get("_exposure_recorded"))
            words_for_log = state.get("last_sentence_words") or [] if exposure_recorded else []
            await repos.log.append(
                "ai",
                ai_msg.content,
                words_used=words_for_log,
                target_words=[{
                    "word": state["target_word"],
                    "context": state.get("target_context") or "",
                }] if state.get("target_word") else [],
                turn_id=turn_id,
            )

        await repos.profile.update(total_turns=turn_id)

        if turn_id % self.config.summary.interval_turns == 0:
            task = asyncio.create_task(self._run_summary_safe(repos))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        return {}

    async def _run_summary_safe(self, repos: Repos) -> None:
        try:
            await run_profile_summary(self.llm_smart, repos)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"background summary failed: {e}")

    async def aclose(self, timeout: float = 2.0) -> None:
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

    async def compile(self, builder: StateGraph, checkpointer=None) -> CompiledStateGraph:
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

    async def _route_after_init_state(self, state: BTPKetState, config: RunnableConfig) -> dict:
        return {}

    async def _passthrough(self, state: BTPKetState, config: RunnableConfig) -> dict:
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


async def build_agent(llm_flash, llm_smart, db, checkpointer=None) -> CompiledStateGraph:
    cfg = load_config()
    agent = KETPartnerAgent(llm_flash, llm_smart, cfg)
    builder = StateGraph(BTPKetState)
    graph = await agent.compile(builder, checkpointer=checkpointer)
    graph.agent = agent  # type: ignore[attr-defined]
    return graph


async def autonomous(info: dict, db_path: str = "ket_partner.db", csv_path: str | None = None) -> CompiledStateGraph:
    db = await init_db(
        db_path,
        csv_path=csv_path,
        default_nickname=info.get("nickname_kid", "宝贝"),
        default_age=info.get("age", 8),
    )
    repos = Repos.for_user(db, "default")
    await repos.log.append_session_start()
    return await build_agent(llm_flash, llm_max, db)
