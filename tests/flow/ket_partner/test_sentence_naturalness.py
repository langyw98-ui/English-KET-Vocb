from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.sentence_domain import (
    _NATURALNESS_SYSTEM as _SYSTEM,
    NaturalnessResult,
    check_naturalness,
)


def _make_llm(return_value: NaturalnessResult):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_check_naturalness_returns_ok():
    llm = _make_llm(NaturalnessResult(ok=True, reason=""))
    result = await check_naturalness(llm, "The cat sleeps on the bed.")
    assert result.ok is True
    assert result.reason == ""
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_naturalness_returns_reject_with_reason():
    llm = _make_llm(NaturalnessResult(ok=False, reason="ice cream does not make noses move"))
    result = await check_naturalness(llm, "The cold ice cream makes my nose move.")
    assert result.ok is False
    assert "nose" in result.reason
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_naturalness_fails_open_on_error():
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await check_naturalness(llm, "anything")
    # Fail-open: judge LLM errors must NOT force extra retries on a non-issue.
    assert result.ok is True
    assert result.reason == ""
    bound.ainvoke.assert_awaited_once()



def test_prompt_covers_three_naturalness_categories():
    """Lock in coverage of the three observed-in-production naturalness
    failure modes so a future prompt refactor can't silently drop one:

    1. Subject-verb-object impossibility ("ice cream makes my nose move").
    2. Semantic redundancy / tautology ("wet water", "cold ice").
    3. Collocation errors ("moves the water off its hair").

    Each was a real production miss that prompted a prompt expansion.
    """
    assert "ice cream does not make noses move" in _SYSTEM, "category 1 (impossibility) example must be present"
    assert "wet water" in _SYSTEM, "category 2 (redundancy) example must be present"
    assert "Collocation" in _SYSTEM or "collocation" in _SYSTEM, "category 3 (collocation) must be named"
