from flow.ket_partner.persistence import KETPartnerRepos
from flow.ket_partner.state import BTPKetState


async def apply_mastery_updates(state: BTPKetState, repos: KETPartnerRepos) -> None:
    intent = state.get("intent")
    if intent == "translation":
        # wrong_words may contain non-KET entries (kept by
        # evaluate_translation_node for DISPLAY feedback). Stats are
        # KET-only: iterate last_sentence_words and check set membership,
        # so non-KET entries in `wrong` are never matched and never reach
        # apply_delta — no phantom stats rows.
        last_words = state.get("last_sentence_words") or []
        target = state.get("last_target_word")
        target_ctx = state.get("last_target_context") or ""
        wrong = {w["word"] for w in state.get("wrong_words") or []}
        # Sentence-level verdict: when overall_correct=False and the per-word
        # check found no misaligned words, the kid still got the sentence
        # structurally wrong (most often by ADDING content with no English
        # source, e.g. "玩球" for "play"). Rewarding every word with +1 here
        # would treat a wrong turn as a right one. Give all words delta=0
        # (neutral — no reward, no punishment, since no specific word is to
        # blame). When wrong_words is non-empty, the structural error doesn't
        # change per-word accounting: misaligned words still get -1, correctly
        # aligned neighbors still get +1.
        overall_correct = state.get("overall_correct")
        neutral_all = (not wrong) and (overall_correct is False)
        if neutral_all:
            return
        for w in last_words:
            # Spec §4.3: target uses real context, scaffolding uses "".
            ctx = target_ctx if w == target else ""
            delta = -1 if w in wrong else 1
            await repos.stats.apply_delta(w, context=ctx, delta=delta, exposed=False)
    elif intent == "idk":
        target = state.get("last_target_word")
        if target:
            target_ctx = state.get("last_target_context") or ""
            await repos.stats.apply_delta(target, context=target_ctx, delta=-1, exposed=False)
    elif intent == "asks_meaning":
        asked = state.get("asked_word")
        if asked:
            # Spec §5.1: token-based lookup, sense unknown — use any_context.
            # Old code passed a WordRef to apply_delta and SQLite rejected it;
            # get_ket_word_any_context returns the canonical str via .word.
            target = state.get("last_target_word")
            target_ctx = state.get("last_target_context") or ""
            wr = await repos.vocab.get_ket_word_any_context(asked)
            if wr:
                # Compare canonical form (wr.word) to target, not the raw
                # asked string — kid may have typed "i" while target is the
                # canonical "I"; both should map to the target's context.
                ctx = target_ctx if wr.word == target else ""
                await repos.stats.apply_delta(wr.word, context=ctx, delta=-1, exposed=False)
    # off_topic / non_compliant: no-op
