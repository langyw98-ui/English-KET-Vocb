from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from flow.common import logger

_SYSTEM = """You judge whether ONE English sentence for a {age}-year-old Chinese kid is NATURAL and makes real-world sense.

Accept (ok=true) if the sentence describes a plausible situation, even if playful or imaginative.
Reject (ok=false) if any of the following apply:

1. Subject-verb-object combinations violate how things actually behave AND the sentence is not a coherent fantasy.
   - Reject: "The cold ice cream makes my nose move." — ice cream does not make noses move.
   - Reject: "The book sings a loud song." — books do not sing.

2. Semantic redundancy / tautology — an adjective or modifier that is inherent to the noun it modifies.
   - Reject: "The dog moves the wet water off its hair." — water is inherently wet; "wet water" is not natural English.
   - Reject: "The cold ice is in the cup." — ice is inherently cold.
   - Reject: "The hot fire burns the wood." — fire is inherently hot.
   - Accept: "The dog shakes the water off its fur." — natural.
   - Accept: "The ice in the cup is melting." — natural.

3. Collocation errors — word combinations that are grammatically valid but native speakers would never use.
   - Reject: "The dog moves the water off its hair." — English speakers say "shakes off" or "dries", not "moves off"; dogs have "fur" not "hair".

Accept examples (natural):
- "The cold ice cream makes my teeth hurt."
- "The funny cat rested in my bed."
- "The monkey eats a yellow banana."
- "The little bird sings a happy song."

When rejecting, `reason` must be ONE short sentence identifying which category (1, 2, or 3) applies and why.
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
    except Exception as e:  # noqa: BLE001
        # Fail-open: if the judge LLM errors, accept the sentence rather than
        # forcing extra retries (and burning budget) on a non-issue.
        logger.warning(f"check_naturalness failed: {e}; accepting by default")
        return NaturalnessResult(ok=True, reason="")
