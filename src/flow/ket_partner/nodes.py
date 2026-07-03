from flow.ket_partner.db import Repos


async def apply_mastery_updates(state: dict, repos: Repos) -> None:
    intent = state.get("intent")
    if intent == "translation":
        # wrong_words is a list of dicts: [{"word": "eat", ...}, ...].
        # It's already been filtered to last_sentence_words subset by
        # evaluate_translation_node, so all entries are KET canonical forms.
        last_words = state.get("last_sentence_words") or []
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
            delta = -1 if w in wrong else 1
            await repos.stats.apply_delta(w, delta=delta, exposed=False)
    elif intent == "idk":
        target = state.get("last_target_word")
        if target:
            await repos.stats.apply_delta(target, delta=-1, exposed=False)
    elif intent == "asks_meaning":
        asked = state.get("asked_word")
        if asked:
            # Use canonical form so the deduction hits the same row that
            # exposure tracking writes to (e.g., kid asks about "I",
            # canonical lookup returns "I" — both reads and writes use "I").
            canonical = await repos.vocab.get_ket_word(asked)
            if canonical:
                await repos.stats.apply_delta(canonical, delta=-1, exposed=False)
    # off_topic / non_compliant: no-op


def format_output_text(state: dict, new_sentence: str) -> str:
    intent = state.get("intent")
    lines = []

    if intent == "translation":
        wrong = state.get("wrong_words") or []
        sentence_t = state.get("sentence_translation", "")
        overall_correct = state.get("overall_correct")
        if wrong:
            if sentence_t:
                lines.append(f"正确翻译：{sentence_t}")
            lines.append("你的翻译有误:")
            for entry in wrong:
                word = entry.get("word", "?")
                correct = entry.get("correct_translation", "?")
                # kid_translation may be empty when the kid OMITTED the word
                # entirely — that's still a real error and must be surfaced,
                # otherwise "你的翻译有误:" renders with no items below it.
                lines.append(f" {word} 的意思是：{correct}")
        elif overall_correct is False:
            # Per-word check found nothing wrong, but the evaluator still
            # flagged the sentence (kid added content with no English source,
            # distorted the meaning, etc.). Without this branch the kid would
            # see no feedback at all — no wrong_words, no correct translation.
            if sentence_t:
                lines.append(f"正确翻译：{sentence_t}")
            lines.append("你的翻译和原句意思有些偏差。")
    elif intent == "idk":
        sentence_t = state.get("sentence_translation", "")
        if sentence_t:
            lines.append(f"正确翻译：{sentence_t}")

    lines.append("请把这句译成中文:")
    lines.append(f'"{new_sentence}"')
    # If generate_sentence_node accepted a sentence with non-KET words,
    # annotate them so the kid can still translate. Skipped on non-generate
    # turns where annotations is None.
    for ann in state.get("non_ket_annotations") or []:
        word = ann.get("word", "?")
        meaning = ann.get("meaning", "")
        if meaning:
            lines.append(f"{word} 的意思是：{meaning}")
    return "\n".join(lines)
