from typing import Literal, Optional

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from flow.common import logger

_SYSTEM = """You classify a Chinese kid's input as one of these intents:
- translation: kid attempts to translate the English sentence
- idk: kid explicitly says "I don't know" / "我不会" / "idk" / "不知道"
- asks_meaning: kid asks what a specific word means
- off_topic: kid chats about something else entirely
- non_compliant: inappropriate / unsafe content

If asks_meaning, extract the asked_word (lowercase English). Otherwise asked_word is null.
"""


class IntentClassification(BaseModel):
    intent: Literal["translation", "idk", "asks_meaning", "off_topic", "non_compliant"]
    asked_word: Optional[str] = Field(default=None)


async def classify_intent(llm, last_english_sentence: Optional[str], kid_input: str) -> IntentClassification:
    structured = llm.with_structured_output(IntentClassification, method="function_calling")
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"English sentence: {last_english_sentence or '(none)'}"),
        HumanMessage(content=f"Kid's input: {kid_input}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:
        logger.warning(f"classify_intent failed: {e}; defaulting to translation")
        return IntentClassification(intent="translation", asked_word=None)
