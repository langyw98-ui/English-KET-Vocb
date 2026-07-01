from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from flow.common import logger

_SYSTEM = """You look up the Chinese meaning of an English word as used in a specific sentence.
Give a concise, kid-friendly Chinese meaning (1-5 characters preferred). Output only the meaning.
"""


class WordMeaning(BaseModel):
    meaning: str


async def lookup_word_meaning(llm, sentence: str, word: str) -> WordMeaning:
    structured = llm.with_structured_output(WordMeaning, method="function_calling")
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Sentence: {sentence}"),
        HumanMessage(content=f"Word: {word}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:
        logger.warning(f"lookup_word_meaning failed: {e}")
        return WordMeaning(meaning=f"({word} 词义查询失败)")
