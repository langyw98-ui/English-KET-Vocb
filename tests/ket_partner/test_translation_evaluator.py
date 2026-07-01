from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.translation_evaluator import (
    TranslationEval,
    evaluate_translation,
)


def _make_llm(return_value: TranslationEval):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_evaluate_finds_wrong_word():
    expected = TranslationEval(
        wrong_words=["cat"],
        correct_meanings={"cat": "猫"},
    )
    llm = _make_llm(expected)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只狗很大",
    )
    assert result.wrong_words == ["cat"]
    assert result.correct_meanings["cat"] == "猫"


@pytest.mark.asyncio
async def test_evaluate_no_wrong_words():
    expected = TranslationEval(wrong_words=[], correct_meanings={})
    llm = _make_llm(expected)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只猫很大",
    )
    assert result.wrong_words == []


@pytest.mark.asyncio
async def test_evaluate_fallback_on_error():
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只猫很大",
    )
    assert result.wrong_words == []
