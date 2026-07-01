from typing import Dict, List

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger

_SYSTEM = """You evaluate a Chinese kid's translation of an English sentence.

Rules:
- Be lenient with synonyms (猫 = 猫咪 = 猫咪儿, 大 = 巨大).
- Function words (the / a / is / on / etc.) are often omitted or flexibly translated in Chinese. Mark them correct unless clearly wrong.
- wrong_words MUST be in the provided sentence words list. Do not invent words.
- For each wrong word, give its correct Chinese meaning in this sentence's context.
"""


class TranslationEval(BaseModel):
    wrong_words: List[str] = Field(default_factory=list)
    correct_meanings: Dict[str, str] = Field(default_factory=dict)


async def evaluate_translation(
    llm,
    sentence: str,
    words: List[str],
    target: str,
    kid_input: str,
) -> TranslationEval:
    structured = llm.with_structured_output(TranslationEval)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"English sentence: {sentence}"),
        HumanMessage(content=f"Sentence words to check: {words}"),
        HumanMessage(content=f"Target word being tested: {target}"),
        HumanMessage(content=f"Kid's Chinese translation: {kid_input}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:
        logger.warning(f"evaluate_translation failed: {e}; defaulting to no wrong words")
        return TranslationEval(wrong_words=[], correct_meanings={})
