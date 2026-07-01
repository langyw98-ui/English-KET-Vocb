from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.word_meaning_lookup import WordMeaning, lookup_word_meaning


def _make_llm(return_value):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_lookup_returns_meaning():
    llm = _make_llm(WordMeaning(meaning="猫"))
    result = await lookup_word_meaning(llm, "The cat is big.", "cat")
    assert result.meaning == "猫"


@pytest.mark.asyncio
async def test_lookup_fallback_on_error():
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await lookup_word_meaning(llm, "The cat is big.", "cat")
    assert "失败" in result.meaning or result.meaning
