import asyncio
from typing import Optional

from langchain.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from flow.agent import memory
from flow.common import logger, llm_flash, llm_max
from flow.ket_partner.config import load_config
from flow.ket_partner.db import Repos, init_db
from flow.ket_partner.graph import route_after_init, route_by_intent
from flow.ket_partner.input_classifier import IntentClassification, classify_intent
from flow.ket_partner.nodes import apply_mastery_updates, format_output_text
from flow.ket_partner.profile_summarizer import run_profile_summary
from flow.ket_partner.sentence_generator import generate_sentence
from flow.ket_partner.sentence_naturalness import check_naturalness
from flow.ket_partner.sentence_validator import validate_sentence
from flow.ket_partner.state import BTPKetState
from flow.ket_partner.translation_evaluator import TranslationEval, evaluate_translation
from flow.ket_partner.vocab_selector import select_target_word
from flow.ket_partner.word_meaning_lookup import WordMeaning, lookup_sentence_translation, lookup_word_meaning, lookup_word_meanings


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
        logger.debug(f"evaluate_translation_node: {result}")
        # Filter to last_sentence_words subset. This is the safety net for:
        # 1. Non-KET words — when generate_sentence_node exhausts retries and
        #    accepts a draft with non_ket_words, those words appear in the
        #    displayed sentence. The LLM may flag them. Without this filter
        #    they'd reach apply_delta and create phantom stats rows.
        # 2. LLM using wrong case / inflected forms — canonical lookup maps
        #    "Eat" → "eat" so it matches the canonical form in last_words.
        # 3. LLM inventing words not in the sentence — silently dropped.
        # 4. LLM emitting the same word twice (observed in production: the
        #    LLM flagged `snow` with two contradictory meanings). Dedup by
        #    word — first entry wins.
        # 5. LLM flagging a word whose kid_translation EXACTLY matches its
        #    correct_translation (both non-empty). The evaluator prompt's
        #    rule 3 forbids this, but when the kid's overall translation is
        #    mostly wrong the LLM occasionally contaminates correct neighbors
        #    anyway. A kid who wrote "我" for "I" (correct_translation "我")
        #    is right by definition — drop it.
        last_words_set = set(state.get("last_sentence_words") or [])
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
            # Drop entries the evaluator couldn't give a clear Chinese
            # meaning for. Common for function words (be/will/do/to) that
            # fold into the sentence structure and have no single Chinese
            # equivalent. Without this filter format_output renders an empty
            # "be 的意思是：". The sentence-level overall_correct flag still
            # captures that the kid's translation was off.
            if not entry.correct_translation:
                logger.debug(
                    f"evaluate_translation_node: dropping '{entry.word}' — "
                    f"empty correct_translation (likely a function word with no single Chinese equivalent)"
                )
                continue
            if entry.word in last_words_set:
                key = entry.word
            else:
                canonical = await self.repos.vocab.get_ket_word(entry.word)
                if canonical and canonical in last_words_set:
                    entry = entry.model_copy(update={"word": canonical})
                    key = canonical
                else:
                    logger.debug(f"evaluate_translation_node: dropping non-KET word '{entry.word}'")
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
        profile = await self.repos.profile.get()

        sentence, result, final_target = await self._generate_with_fallback(
            state["target_word"], avoid_words, avoid_sentences, age, profile
        )
        target = final_target

        self._recent_sentences.append(sentence)
        # Multi-word / non-alpha target patch. The validator tokenizes with
        # [A-Za-z']+, so targets like "MP3 player", "T-shirt", "ice cream" get
        # split (or partially dropped — "MP3" → "MP" silently skipped as a
        # proper noun). Force-include the target when it actually appears in
        # the sentence so stats tracking marks it 'learning' and downstream
        # filters (which use last_sentence_words) recognize the same lexical
        # unit the kid was asked about. (Note: the retry loop's target_split
        # check above guarantees the contiguous presence for multi-word
        # targets by the time we reach here; single-word targets always
        # pass the substring test trivially.)
        if (
            target
            and target not in result.words_used
            and target.lower() in sentence.lower()
        ):
            result.words_used.append(target)
            # The validator also tracked the target's trailing constituent as
            # a standalone scaffolding word (e.g. "player" from "MP3 player").
            # Drop it — in that sentence the word is part of the target
            # phrase, not an independent lexical unit. Simple whitespace-split
            # covers space-separated phrases ("MP3 player", "ice cream").
            # Hyphenated / period targets ("T-shirt", "a.m.") still leak their
            # alphabetic tail; rare in KET vocab, accepted as a known limit.
            constituents = {c.lower() for c in target.split()}
            result.words_used = [
                w for w in result.words_used
                if w == target or w.lower() not in constituents
            ]
            # Also strip constituents from non_ket_words. When the target is
            # a multi-word KET entry like "lie down", the validator tokenizes
            # it into ["lie", "down"]. "down" matches a KET entry, but "lie"
            # might not (lie-alone isn't in KET vocab) — so "lie" lands in
            # non_ket_words and gets separately annotated, even though in
            # THIS sentence it's part of the target phrase. Same whitespace-
            # split limitation as above (hyphenated targets still leak).
            result.non_ket_words = [
                w for w in result.non_ket_words
                if w.lower() not in constituents
            ]
            logger.debug(
                f"multi-word target patch: added '{target}', "
                f"final words_used={result.words_used}, "
                f"non_ket_words={result.non_ket_words}"
            )
        # Append to _recent_scaffolding AFTER the patch — otherwise the
        # in-place mutation + reassignment leaves the outer list pointing at
        # an intermediate state that contains BOTH "player" AND "MP3 player",
        # which then leaks into the next turn's avoid_words list.
        self._recent_scaffolding.append(result.words_used)
        # Per spec §11.9, exposed_count is incremented once per word in the
        # NEW sentence. Do this here (on the generate path) so non-generate
        # turns do not re-count the prior sentence's words. Set the flag so
        # persist_turn_node knows the increment is already done and skips its
        # own increment.
        # Mark the target distinctly: its first exposure makes it 'learning'
        # (active practice target); scaffolding words stay 'exposed' until
        # they're selected as target in a future turn. Without this flag, all
        # words would pool into 'exposed' and refill_mode / oldest_learning_word
        # would treat passive scaffolding as if it were target practice.
        for w in result.words_used:
            await self.repos.stats.increment_exposed(w, is_target=(w == target))
        # If the accepted sentence still has non-KET words, look up their
        # context meanings so format_output_node can annotate them for the kid.
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
        # If the fallback switched the target word, propagate the new target
        # to state so persist_turn_node logs the correct target_words and
        # downstream nodes see the word that was actually practiced.
        if target != state["target_word"]:
            update["target_word"] = target
        return update

    async def _validate_and_categorize(self, sentence: str, target: str, age: int) -> dict:
        """Validate a sentence and categorize its failure (if any).

        Centralizes the validate + classify logic that the retry loop uses on
        every attempt. Returns a dict with:
        - result: ValidationResult from validate_sentence
        - passed: True if the sentence passes all gates (KET + dedup + target
          contiguity + naturalness)
        - reason_kind: None if passed, else one of "naturalness",
          "non_ket_overflow", "duplicate", "target_split"
        - reason_detail: human-readable explanation (empty if passed)
        - non_ket_words / non_ket_count: from the ValidationResult
        - is_duplicate / is_target_split: the structural gate flags
        """
        result = await validate_sentence(sentence, self.repos)
        is_duplicate = sentence in self._recent_sentences
        non_ket_count = len(result.non_ket_words)
        is_target_split = (
            bool(target)
            and " " in target.strip()
            and target.lower() not in sentence.lower()
        )

        passed = False
        reason_kind = None
        reason_detail = ""

        if non_ket_count <= 1 and not is_duplicate and not is_target_split:
            if non_ket_count == 0:
                # Only run the expensive naturalness LLM check on sentences
                # that fully passed the cheap KET gate — otherwise we'd
                # burn budget judging sentences we'll regen anyway.
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
                # non_ket_count == 1: accept as-is, skip naturalness (no point
                # judging a sentence the policy already chose to annotate).
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
        avoid_words: list,
        avoid_sentences: list,
        age: int,
        profile: dict,
    ):
        """Generate a sentence with retry + smart fallback.

        Returns (sentence, validation_result, final_target).
        `final_target` differs from `initial_target` when a word switch occurred.

        Retry policy (per user spec):
        - Each attempt is validated; on failure the (sentence, reason) is
          appended to `attempts` and fed back to the next regen's prompt
          via prior_attempts — the LLM sees ALL prior failures, not just
          the latest.
        - After retries exhaust:
          * If any attempt had non_ket_overflow (>1 non-KET words), pick
            the one with the FEWEST non-KET words (tie → latest) and accept
            it — non-KET overflow is a tolerable failure (natural English
            with some unknown words the kid can still translate via the
            annotations).
          * Elif ALL attempts were naturalness fails AND we haven't yet
            switched the target word, call select_target_word again and
            run a full retry cycle with the new word. Bounded to ONE switch
            so we can't loop forever.
          * Otherwise (mixed failures, or word already switched and still
            failing), accept the final draft.
        """
        target = initial_target
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
                )

            sentence = await _regen()
            result = None

            for _ in range(self.config.validate_retry_limit):
                check = await self._validate_and_categorize(sentence, target, age)
                result = check["result"]
                if check["passed"]:
                    return sentence, result, target  # SUCCESS
                attempts.append({
                    "sentence": sentence,
                    "reason_kind": check["reason_kind"],
                    "reason_detail": check["reason_detail"],
                    "non_ket_words": check["non_ket_words"],
                    "non_ket_count": check["non_ket_count"],
                })
                # Accumulate non-KET words across retries — the LLM frequently
                # reuses the same non-KET word (e.g. "blocks" for "build")
                # because the target naturally co-occurs with it. Surfacing
                # them as off-limits forces different scaffolding.
                for w in check["non_ket_words"]:
                    if w not in seen_non_ket_words:
                        seen_non_ket_words.append(w)
                sentence = await _regen()

            # Retries exhausted — validate the final draft and record its
            # outcome (the for-loop's last iteration did `sentence = _regen()`
            # but never validated that new sentence).
            check = await self._validate_and_categorize(sentence, target, age)
            result = check["result"]
            if check["passed"]:
                return sentence, result, target  # SUCCESS (final draft passed)
            attempts.append({
                "sentence": sentence,
                "reason_kind": check["reason_kind"],
                "reason_detail": check["reason_detail"],
                "non_ket_words": check["non_ket_words"],
                "non_ket_count": check["non_ket_count"],
            })

            # Smart fallback.
            overflow_attempts = [a for a in attempts if a["reason_kind"] == "non_ket_overflow"]
            all_naturalness = bool(attempts) and all(
                a["reason_kind"] == "naturalness" for a in attempts
            )

            if overflow_attempts:
                # Pick the one with fewest non-KET words (least bad). Tie →
                # latest attempt: min() on reversed() returns the last element
                # among those tied at the minimum, so the kid sees the most
                # recent LLM output (which has seen the most prior feedback).
                best = min(reversed(overflow_attempts), key=lambda a: a["non_ket_count"])
                sentence = best["sentence"]
                result = await validate_sentence(sentence, self.repos)
                logger.warning(
                    f"sentence validation: accepting non-KET overflow draft after "
                    f"{len(attempts)} attempts (non_ket_count={len(result.non_ket_words)}); "
                    f"sentence={sentence!r}"
                )
                return sentence, result, target
            elif all_naturalness and not word_switched:
                # All naturalness fails — the LLM can't produce a natural
                # sentence for this target. Switch to a different word and
                # give it a fresh retry cycle.
                logger.info(
                    f"all {len(attempts)} attempts failed on naturalness; "
                    f"switching target word from '{target}'"
                )
                new_word = await select_target_word(self.repos, profile, self.config)
                if new_word == target:
                    logger.warning(
                        f"could not find a different target word; "
                        f"accepting final draft: {sentence!r}"
                    )
                    return sentence, result, target
                target = new_word
                word_switched = True
                continue  # restart while loop with new word, fresh attempts
            else:
                # Mixed failures (duplicate/target_split/naturalness mix
                # without overflow) OR word already switched and still
                # failing — accept the final draft.
                reasons = []
                if check["non_ket_count"] > 1:
                    reasons.append(f"{check['non_ket_count']} non-KET word(s): {check['non_ket_words']}")
                if check["is_duplicate"]:
                    reasons.append("duplicate of a recent sentence")
                if check["is_target_split"]:
                    reasons.append(f"multi-word target '{target}' split apart")
                if not reasons:
                    # naturalness was the reason (already in attempts)
                    reasons.append(f"naturalness: {check['reason_detail']}")
                logger.warning(
                    f"sentence validation failed after {len(attempts)} attempts; "
                    f"accepting current draft — reasons: {('; '.join(reasons)) or 'unknown'}; "
                    f"sentence={sentence!r}"
                )
                return sentence, result, target

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
        last = state.get("last_english_sentence") or ""
        meaning = state.get("asked_word_meaning", "")
        text = f'"{asked}" 的意思是「{meaning}」。\n让我们继续吧，{last}。'
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
    return await build_agent(llm_flash, llm_max, repos, info)
