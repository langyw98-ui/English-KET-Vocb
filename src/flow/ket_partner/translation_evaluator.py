from typing import List

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger

_SYSTEM = """You evaluate a Chinese kid's translation of an English sentence.

METHOD (do this silently before filling the schema):
1. ALIGN each English word in "Sentence words to check" to the kid's Chinese characters. Chinese word order differs from English — verbs often sit at the end or middle (e.g. "The cat slipped on the ice" → "猫 在冰上 滑倒"). Map each English word to the kid's characters that mean the SAME thing.
2. A word is WRONG only if its aligned Chinese characters are missing or carry a different meaning. Synonyms are OK (猫=猫咪, 大=巨大).
3. A word is NOT wrong if the kid's output contains its correct meaning — even when OTHER words in the sentence are wrong. This is the most common error: do not let one wrong word contaminate the verdict on its neighbors.

Then fill the schema:
- correct_translation: full correct Chinese translation of the whole sentence in natural, kid-friendly Chinese.
- wrong_words: list of words the kid got wrong.

For each wrong word:
- word: the EXACT form from "Sentence words to check" (no inflections: "eat" not "eats", "cat" not "cats").
- kid_translation: the Chinese characters the kid wrote for THIS word. Empty string if the kid omitted it entirely.
- correct_translation: the correct Chinese meaning in THIS sentence's context.
- contrast: optional short explanation when there's confusion (e.g. "在 means 'at', not 'eat'"). Empty string if not needed.

WORKED EXAMPLE (do not skip this):
- Sentence: "The cat slipped on the ice."
- Kid's translation: "猫在冰上飞"  (fly)
- Alignment: cat→猫 (correct); the→(article, Chinese omits, OK); slipped→飞 (WRONG: kid wrote "fly", should be "slip/slide"); on→在...上 (correct); ice→冰 (correct).
- Correct wrong_words: [{word: "slipped", kid_translation: "飞", correct_translation: "滑倒"}].
- DO NOT flag "ice" — the kid wrote "冰" which is correct, even though the overall sentence has a wrong verb nearby.

If everything is correct, return an empty wrong_words list.
"""


class WrongWord(BaseModel):
    word: str
    kid_translation: str = ""
    correct_translation: str
    contrast: str = ""


class TranslationEval(BaseModel):
    correct_translation: str
    wrong_words: List[WrongWord] = Field(default_factory=list)


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
        return TranslationEval(correct_translation="", wrong_words=[])
