from __future__ import annotations

import asyncio

import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from flow.common import LlmService, logger
from flow.ket_partner.config import KetConfig
from flow.ket_partner.dialogue_domain import (
    WrongWord,
    classify_intent,
    evaluate_translation,
    format_output_text,
    run_profile_summary,
)
from flow.ket_partner.persistence import KETPartnerRepos, get_repos
from flow.ket_partner.sentence_domain import (
    _tokenize,
    apply_multiword_target_patch,
    generate_with_fallback,
)
from flow.ket_partner.state import BTPKetState
from flow.ket_partner.vocab_domain import (
    apply_mastery_updates,
    lookup_sentence_translation,
    lookup_word_meaning,
    lookup_word_meanings,
    select_target_word,
)

# _run_summary_safe 后台任务的可重试外部失败类型。
# 严格按 CLAUDE.md §1.5:只含具体外部失败,不含 ValueError/AttributeError/TypeError
# 等代码 bug 类型——那些必须直接暴露被测试捕获。
_LLM_RETRYABLE: tuple[type[BaseException], ...] = (
    openai.APIError,          # openai SDK 的所有 API 异常基类(APITimeoutError/APIConnectionError/RateLimitError 等)
    asyncio.TimeoutError,     # asyncio.wait_for 超时
    ValidationError,          # pydantic Schema 校验失败(LLM 返回畸形结构)
)


class KETPartnerAgent:
    """KET Partner 对话 agent。

    LLM 通过 LlmService Protocol 注入(Phase 2 DI 重构),业务代码不直接
    持有 BaseChatModel,而是依赖 service.smart / service.flash 属性,
    便于测试用 mock service 替换。

    - _llm_service: 仅 __init__ 在构造时写;其他位置只读。
    - config: 仅 __init__ 在构造时写;其他位置只读。
    - _bg_tasks: 仅 _run_summary_safe 创建任务时 add,aclose 时迭代清理;其他位置只读。
    """

    def __init__(self, llm_service: LlmService, config: KetConfig) -> None:
        self._llm_service = llm_service
        self.config = config
        self._bg_tasks: set[asyncio.Task] = set()

    @property
    def llm_smart(self) -> BaseChatModel:
        return self._llm_service.smart

    @property
    def llm_flash(self) -> BaseChatModel:
        return self._llm_service.flash

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
                    # P2 #22 修复:不修改循环变量 entry,改为直接构造新 WrongWord 实例。
                    entry = WrongWord(
                        word=wr.word,
                        kid_translation=entry.kid_translation,
                        correct_translation=entry.correct_translation,
                        contrast=entry.contrast,
                    )
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
        except _LLM_RETRYABLE as e:
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
