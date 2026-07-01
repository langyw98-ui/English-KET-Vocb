from flow.common import logger
from flow.ket_partner.db import Repos


async def apply_mastery_updates(state: dict, repos: Repos) -> None:
    intent = state.get("intent")
    if intent == "translation":
        last_words = state.get("last_sentence_words") or []
        wrong = set(state.get("wrong_words") or [])
        for w in last_words:
            delta = -1 if w in wrong else 1
            await repos.stats.apply_delta(w, delta=delta, exposed=False)
    elif intent == "idk":
        target = state.get("last_target_word")
        if target:
            await repos.stats.apply_delta(target, delta=-1, exposed=False)
    elif intent == "asks_meaning":
        asked = state.get("asked_word")
        if asked and await repos.vocab.is_ket_word(asked):
            await repos.stats.apply_delta(asked, delta=-1, exposed=False)
    # off_topic / non_compliant: no-op


def format_output_text(state: dict, new_sentence: str) -> str:
    intent = state.get("intent")
    lines = []

    if intent == "translation":
        wrong = state.get("wrong_words") or []
        meanings = state.get("correct_meanings") or {}
        if wrong:
            lines.append("📖 上句解析:")
            for w in wrong:
                m = meanings.get(w, "?")
                lines.append(f"  - {w} →「{m}」")
            target = state.get("last_target_word")
            if target and target not in wrong:
                lines.append(f"  - {target} → ✓")
            lines.append("")
    elif intent == "idk":
        target = state.get("last_target_word")
        meaning = state.get("target_word_meaning", "")
        lines.append("📖 上句学习:")
        lines.append(f"  - {target} 的意思是「{meaning}」")
        lines.append("")

    lines.append("🔤 请把这句译成中文:")
    lines.append(f'"{new_sentence}"')
    return "\n".join(lines)
