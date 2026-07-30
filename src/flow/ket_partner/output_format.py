from flow.ket_partner.state import BTPKetState


def format_output_text(state: BTPKetState, new_sentence: str) -> str:
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
