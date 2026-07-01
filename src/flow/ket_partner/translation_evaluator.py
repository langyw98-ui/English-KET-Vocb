from typing import Dict, List

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger

_SYSTEM = """You evaluate a Chinese kid's translation of an English sentence.

Rules:
- Be lenient with synonyms (猫 = 猫咪 = 猫咪儿, 大 = 巨大).
- wrong_words MUST be in the provided sentence words list. Do not invent content words.
- For each wrong content word, give its correct Chinese meaning in this sentence's context.
- PREPOSITIONS AND SPATIAL PARTICLES (in/on/at/under/over/above/below/by/with/from/into/out of/off/up/down/around/through) are high-confusion points for Chinese kids. Read the kid's translation carefully and check each preposition:
  * If the kid used the WRONG Chinese equivalent (e.g. "in" translated as 上 instead of 里), add it to function_word_errors.
  * Do NOT flag a preposition just because the kid omitted it (Chinese often drops prepositions) — only flag when the kid wrote a Chinese word that means a DIFFERENT preposition.
- Articles (the / a / an) and basic copulas (is / are / was / were) — be lenient, do not flag.
- If everything is correct, return empty lists/dicts.
"""


class FunctionWordError(BaseModel):
    word: str
    kid_translation: str
    correct_translation: str
    contrast: str = ""


class TranslationEval(BaseModel):
    wrong_words: List[str] = Field(default_factory=list)
    correct_meanings: Dict[str, str] = Field(default_factory=dict)
    function_word_errors: List[FunctionWordError] = Field(default_factory=list)


async def evaluate_translation(
    llm,
    sentence: str,
    words: List[str],
    target: str,
    kid_input: str,
) -> TranslationEval:
    structured = llm.with_structured_output(TranslationEval, method="function_calling")
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
        return TranslationEval(wrong_words=[], correct_meanings={}, function_word_errors=[])
