from langchain.messages import HumanMessage, SystemMessage

from flow.common import logger

_SYSTEM = """You write ONE English sentence for a {age}-year-old Chinese kid to translate.

Constraints:
- {min_words}-{max_words} words, single sentence.
- Naturally include the target word: "{target}".
- Vary scaffolding words. Don't reuse words from recent sentences: {recent}.
- Prefer words the kid has likely mastered.
- All words must be from KET vocabulary (will be validated).
- Fun & memorable (silly situations OK).
- NO emoji, NO Chinese.

Output: just the English sentence, nothing else.
"""


async def generate_sentence(
    llm,
    target: str,
    recent_scaffolding: list,
    age: int,
    min_words: int,
    max_words: int,
) -> str:
    creative = llm.bind(temperature=0.8)
    system_text = _SYSTEM.format(
        age=age,
        min_words=min_words,
        max_words=max_words,
        target=target,
        recent=", ".join(recent_scaffolding) or "(none yet)",
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
