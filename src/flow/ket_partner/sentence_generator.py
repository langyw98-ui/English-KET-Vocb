from langchain.messages import HumanMessage, SystemMessage

from flow.common import logger

_SYSTEM = """You write ONE English sentence for a {age}-year-old Chinese kid to translate.

Constraints:
- {min_words}-{max_words} words, single sentence.
- Naturally include the target word: "{target}".
- Vary scaffolding words. Don't reuse words from recent sentences: {recent}.
- Do NOT output any of these exact sentences (must differ in wording, subject, or scene): {avoid}
- Prefer words the kid has likely mastered.
- All words must be from KET vocabulary (will be validated).
- Must be NATURAL and make real-world sense. Subject-verb-object must reflect how things actually behave. No nonsense like "ice cream makes my nose move" or "the book sings".
- Playful or imaginative situations are fine, but only if internally coherent.
- NO emoji, NO Chinese.
{hint_block}
Output: just the English sentence, nothing else.
"""

_HINT_BLOCK = """
Previous attempt rejected by naturalness check: {hint}
Avoid that specific issue this time.
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
) -> str:
    creative = llm.bind(temperature=0.8)
    avoid_sentences = avoid_sentences or []
    hint_block = _HINT_BLOCK.format(hint=naturalness_hint) if naturalness_hint else ""
    system_text = _SYSTEM.format(
        age=age,
        min_words=min_words,
        max_words=max_words,
        target=target,
        recent=", ".join(recent_scaffolding) or "(none yet)",
        avoid="\n".join(f"  - {s}" for s in avoid_sentences) or "(none yet)",
        hint_block=hint_block,
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


_REWRITE_SYSTEM = """You revise ONE English sentence for a {age}-year-old Chinese kid.

Your job: replace ONLY these words: {replace}.
- Keep the original meaning and sentence structure as much as possible.
- Keep the target word "{target}" in the sentence.
- Replace each listed word with a KET-vocabulary alternative that fits the context.
- Do NOT touch any other word.
- The result must NOT equal any of these recent sentences (change wording if needed): {avoid}
- Length stays {min_words}-{max_words} words.
- NO emoji, NO Chinese.

Original sentence: {original}

Output: just the revised English sentence, nothing else.
"""


async def rewrite_sentence(
    llm,
    original: str,
    replace_words: list,
    target: str,
    age: int,
    min_words: int,
    max_words: int,
    avoid_sentences: list = None,
) -> str:
    """Targeted rewrite: ask the LLM to swap specific non-KET words while
    preserving the rest of the sentence. Used when validation fails on a
    small number of words — far higher hit-rate than a fresh regen.
    """
    creative = llm.bind(temperature=0.6)
    avoid_sentences = avoid_sentences or []
    replace_str = ", ".join(replace_words)
    system_text = _REWRITE_SYSTEM.format(
        age=age,
        replace=replace_str,
        target=target,
        min_words=min_words,
        max_words=max_words,
        original=original,
        avoid="\n".join(f"  - {s}" for s in avoid_sentences) or "(none yet)",
    )
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=f"Replace these words: {replace_str}."),
    ]
    logger.debug(f"rewrite_sentence: {messages}")
    try:
        response = await creative.ainvoke(messages)
        revised = response.content.strip()
        logger.debug(f"rewrite_sentence: {revised}")
        return revised
    except Exception as e:
        logger.warning(f"rewrite_sentence failed: {e}")
        return original
