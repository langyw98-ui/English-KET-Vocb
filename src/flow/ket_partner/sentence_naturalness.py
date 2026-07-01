from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from flow.common import logger

_SYSTEM = """You judge whether ONE English sentence for a {age}-year-old Chinese kid is NATURAL and makes real-world sense.

Accept (ok=true) if the sentence describes a plausible situation, even if playful or imaginative.
Reject (ok=false) if subject-verb-object combinations violate how things actually behave, AND the sentence is not clearly a coherent fantasy.

Reject examples:
- "The cold ice cream makes my nose move." — ice cream does not make noses move; reject.
- "The book sings a loud song." — books do not sing; reject.

Accept examples:
- "The cold ice cream makes my teeth hurt." — plausible; accept.
- "The funny cat rested in my bed." — plausible; accept.
- "The monkey eats a yellow banana." — plausible; accept.
- "The little bird sings a happy song." — plausible; accept.

When rejecting, `reason` must be ONE short sentence explaining why (e.g. "ice cream does not make noses move").
When accepting, leave `reason` empty.

Output only the structured fields.
"""


class NaturalnessResult(BaseModel):
    ok: bool
    reason: str = ""


async def check_naturalness(llm, sentence: str, age: int = 8) -> NaturalnessResult:
    structured = llm.with_structured_output(NaturalnessResult, method="function_calling")
    messages = [
        SystemMessage(content=_SYSTEM.format(age=age)),
        HumanMessage(content=f"Sentence: {sentence}"),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as e:
        # Fail-open: if the judge LLM errors, accept the sentence rather than
        # forcing extra retries (and burning budget) on a non-issue.
        logger.warning(f"check_naturalness failed: {e}; accepting by default")
        return NaturalnessResult(ok=True, reason="")
