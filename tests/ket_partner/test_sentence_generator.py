from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.sentence_generator import generate_sentence


def _make_llm(return_value: str):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=MagicMock(content=return_value))
    llm.bind = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_generate_sentence_returns_string():
    llm = _make_llm("The big cat is sleeping on the bed.")
    result = await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=["dog", "fish"],
        age=8,
        min_words=5,
        max_words=12,
    )
    assert isinstance(result, str)
    assert "cat" in result


@pytest.mark.asyncio
async def test_generate_sentence_handles_empty_recent():
    llm = _make_llm("I see a cat.")
    result = await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    assert "cat" in result
