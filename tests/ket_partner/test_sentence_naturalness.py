from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.sentence_naturalness import NaturalnessResult, check_naturalness


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


@pytest.mark.asyncio
async def test_check_naturalness_returns_reject_with_reason():
    llm = _make_llm(NaturalnessResult(ok=False, reason="ice cream does not make noses move"))
    result = await check_naturalness(llm, "The cold ice cream makes my nose move.")
    assert result.ok is False
    assert "nose" in result.reason


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
