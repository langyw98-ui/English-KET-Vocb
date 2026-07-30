import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from flow.common import logger
from flow.ket_partner.input_classifier import classify_intent
from flow.ket_partner.nodes import apply_mastery_updates, format_output_text
from flow.ket_partner.persistence import KETPartnerRepos, get_repos
from flow.ket_partner.profile_summarizer import run_profile_summary
from flow.ket_partner.sentence_orchestration import (
    apply_multiword_target_patch,
    generate_with_fallback,
)
from flow.ket_partner.sentence_validator import _tokenize
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
        repos = get_repos(config)
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
        repos = get_repos(config)
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
        repos = get_repos(config)
        await apply_mastery_updates(state, repos)
        return {}

    async def select_target_word_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos = get_repos(config)
        profile = await repos.profile.get()
        word_ref = await select_target_word(repos, profile, self.config)
        if word_ref is None:
            return {"target_word": None, "target_context": None}
        return {"target_word": word_ref.word, "target_context": word_ref.context}

    async def generate_sentence_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        repos = get_repos(config)
        user_info: dict = config["configurable"].get("user_info", {})
        age = user_info.get("age", 8)
        window = self.config.variety.recent_window
        avoid_words = [w for sent_words in (await repos.recent.list_recent_scaffolding(window=window)) for w in sent_words]
        avoid_sentences = await repos.recent.list_recent(limit=window)
        profile = await repos.profile.get()

        target = state["target_word"]
        target_ctx = state.get("target_context") or ""

        sentence, result, final_target, final_ctx = await generate_with_fallback(
            self.llm_smart,
            initial_target=state["target_word"],
            initial_context=target_ctx,
            avoid_words=avoid_words,
            avoid_sentences=avoid_sentences,
            age=age,
            profile=profile,
            repos=repos,
            config=self.config,
        )
        target, target_ctx = final_target, final_ctx

        await repos.recent.append(sentence, window=window)

        apply_multiword_target_patch(target, sentence, result)
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
        repos = get_repos(config)
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

    async def _run_summary_safe(self, repos: KETPartnerRepos) -> None:
        try:
            await run_profile_summary(self.llm_smart, repos)
        except (TimeoutError, RuntimeError, ValueError, AttributeError, TypeError) as e:
            logger.warning(f"background summary failed: {e}", exc_info=True)

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
