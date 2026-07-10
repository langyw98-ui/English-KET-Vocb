from langchain.messages import HumanMessage, SystemMessage

from flow.common import logger
from flow.ket_partner.multi_word_target import find_placeholder, has_placeholder

_SYSTEM = """You write ONE English sentence for a {age}-year-old Chinese kid to translate.

Constraints:
- {min_words}-{max_words} words, single sentence.
- Naturally include the target: "{target}".{multi_word_note}
- Vary scaffolding words. Don't reuse words from recent sentences: {recent}.
- Do NOT output any of these exact sentences (must differ in wording, subject, or scene): {avoid}
- Prefer words the kid has likely mastered.
- All words must be from KET vocabulary (will be validated).
- Must be NATURAL and make real-world sense. Subject-verb-object must reflect how things actually behave. No nonsense like "ice cream makes my nose move" or "the book sings".
- Playful or imaginative situations are fine, but only if internally coherent.
- NO emoji, NO Chinese.
{history_block}{non_ket_block}{target_split_block}{context_block}
Output: just the English sentence, nothing else.
"""

_HISTORY_BLOCK = """
Your previous attempts were all rejected. Do NOT repeat the same mistakes — address each rejection's underlying issue (do not just change the subject or surrounding words while keeping the same problematic collocation / word choice):
{history}
"""

_NON_KET_BLOCK = """
Previous attempt(s) used these non-KET words — do NOT use them again, pick KET alternatives instead:
{words}
"""

_TARGET_SPLIT_BLOCK = """
At least one previous attempt SPLIT the target phrase — its words did NOT appear contiguously in the sentence.
The target "{target}" MUST appear as a single inseparable unit, with its words side-by-side in the same order.
Rewrite the sentence so the target phrase is intact.
"""

_CONTEXT_BLOCK = """
Use "{target}" specifically in this sense: {context}.
"""

# Inline note appended to the target line when the target is a multi-word
# phrase. Without this, the LLM treats "CD player" as "use CD and player
# somewhere" and emits sentences like "He puts a CD into the old player." —
# the target phrase never appears contiguously, so stats tracking misses it.
_MULTI_WORD_NOTE = " The target is a MULTI-WORD phrase — its words MUST appear contiguously in the sentence, side-by-side in the same order. Do NOT split them with other words."

# Inline note for placeholder phrases like "give somebody a call" — the
# placeholder token (somebody/someone/something/...) must be REPLACED by a
# concrete noun/pronoun, not kept literally. The multi-word note would
# wrongly instruct the LLM to keep "somebody" verbatim.
_PLACEHOLDER_MULTI_WORD_NOTE = ' The target is a multi-word phrase with a PLACEHOLDER — "{placeholder}" MUST be REPLACED by a concrete noun, name, or pronoun (e.g. him / my mom / the teacher). The OTHER words must appear contiguously and in the same order, with the replacement occupying exactly the placeholder\'s slot. Do NOT keep "{placeholder}" in the sentence.'


def _select_multi_word_note(target: str) -> str:
    if not target or " " not in target.strip():
        return ""
    if has_placeholder(target):
        return _PLACEHOLDER_MULTI_WORD_NOTE.format(placeholder=find_placeholder(target))
    return _MULTI_WORD_NOTE


def _format_history(attempts: list) -> str:
    if not attempts:
        return ""
    lines = []
    for i, a in enumerate(attempts, 1):
        lines.append(f'  {i}. "{a["sentence"]}" — Reason: {a["reason_detail"]}')
    return "\n".join(lines)


async def generate_sentence(
    llm,
    target: str,
    recent_scaffolding: list,
    age: int,
    min_words: int,
    max_words: int,
    avoid_sentences: list = None,
    prior_attempts: list = None,
    avoid_non_ket_words: list = None,
    target_context: str = "",
) -> str:
    creative = llm.bind(temperature=0.8)
    avoid_sentences = avoid_sentences or []
    prior_attempts = prior_attempts or []
    avoid_non_ket_words = avoid_non_ket_words or []
    history_block = _HISTORY_BLOCK.format(history=_format_history(prior_attempts)) if prior_attempts else ""
    non_ket_block = _NON_KET_BLOCK.format(words=", ".join(avoid_non_ket_words)) if avoid_non_ket_words else ""
    multi_word_note = _select_multi_word_note(target)
    # target_split_block fires if ANY prior attempt split the target — the LLM
    # tends to repeat the split pattern across retries when the target is a
    # multi-word phrase, so a single earlier split is enough to warrant the
    # strong reminder on every subsequent regen.
    target_split_block = (
        _TARGET_SPLIT_BLOCK.format(target=target)
        if any(a.get("reason_kind") == "target_split" for a in prior_attempts)
        else ""
    )
    context_block = (
        _CONTEXT_BLOCK.format(target=target, context=target_context)
        if target_context else ""
    )
    system_text = _SYSTEM.format(
        age=age,
        min_words=min_words,
        max_words=max_words,
        target=target,
        recent=", ".join(recent_scaffolding) or "(none yet)",
        avoid="\n".join(f"  - {s}" for s in avoid_sentences) or "(none yet)",
        multi_word_note=multi_word_note,
        history_block=history_block,
        non_ket_block=non_ket_block,
        target_split_block=target_split_block,
        context_block=context_block,
    )
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"Write a sentence using target word '{target}'."),
    ]
    logger.debug(f"generate_sentence: {messages}")
    try:
        response = await creative.ainvoke(messages)
        logger.debug(f"generate_sentence: {response.content.strip()}")
        return response.content.strip()
    except Exception as e:
        logger.warning(f"generate_sentence failed: {e}")
        return f"I see a {target}."
