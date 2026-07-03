from langchain.messages import HumanMessage, SystemMessage

from flow.common import logger

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
{target_split_block}{duplicate_block}{non_ket_block}{hint_block}
Output: just the English sentence, nothing else.
"""

_HINT_BLOCK = """
Previous attempt rejected by naturalness check: {hint}
Avoid that specific issue this time.
"""

_NON_KET_BLOCK = """
Previous attempt(s) used these non-KET words — do NOT use them again, pick KET alternatives instead:
{words}
"""

_DUPLICATE_BLOCK = """
Your previous attempt was a WORD-FOR-WORD DUPLICATE of a recent sentence:
{duplicate}
That exact sentence is forbidden. Do NOT output it again — change the wording, subject, or scene.
"""

# Inline note appended to the target line when the target is a multi-word
# phrase. Without this, the LLM treats "CD player" as "use CD and player
# somewhere" and emits sentences like "He puts a CD into the old player." —
# the target phrase never appears contiguously, so stats tracking misses it.
_MULTI_WORD_NOTE = " The target is a MULTI-WORD phrase — its words MUST appear contiguously in the sentence, side-by-side in the same order. Do NOT split them with other words."

# Retry hint when the previous attempt split the target phrase. Stronger
# than the initial note because the LLM has already failed the constraint.
_TARGET_SPLIT_BLOCK = """
Your previous attempt SPLIT the target phrase — its words did NOT appear contiguously in the sentence.
The target "{target}" MUST appear as a single inseparable unit, with its words side-by-side in the same order.
Rewrite the sentence so the target phrase is intact.
"""


async def generate_sentence(
    llm,
    target: str,
    recent_scaffolding: list,
    age: int,
    min_words: int,
    max_words: int,
    avoid_sentences: list = None,
    naturalness_hint: str = "",
    avoid_non_ket_words: list = None,
    duplicate_sentence: str = "",
    target_split: bool = False,
) -> str:
    creative = llm.bind(temperature=0.8)
    avoid_sentences = avoid_sentences or []
    avoid_non_ket_words = avoid_non_ket_words or []
    hint_block = _HINT_BLOCK.format(hint=naturalness_hint) if naturalness_hint else ""
    non_ket_block = _NON_KET_BLOCK.format(words=", ".join(avoid_non_ket_words)) if avoid_non_ket_words else ""
    duplicate_block = _DUPLICATE_BLOCK.format(duplicate=duplicate_sentence) if duplicate_sentence else ""
    multi_word_note = _MULTI_WORD_NOTE if target and " " in target.strip() else ""
    target_split_block = _TARGET_SPLIT_BLOCK.format(target=target) if target_split else ""
    system_text = _SYSTEM.format(
        age=age,
        min_words=min_words,
        max_words=max_words,
        target=target,
        recent=", ".join(recent_scaffolding) or "(none yet)",
        avoid="\n".join(f"  - {s}" for s in avoid_sentences) or "(none yet)",
        multi_word_note=multi_word_note,
        target_split_block=target_split_block,
        duplicate_block=duplicate_block,
        hint_block=hint_block,
        non_ket_block=non_ket_block,
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
        return f"I see a {target}."  # 兜底
