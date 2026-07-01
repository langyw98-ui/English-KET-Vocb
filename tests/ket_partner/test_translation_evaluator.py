from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.translation_evaluator import (
    TranslationEval,
    WrongWord,
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
        correct_translation="那只猫很大",
        wrong_words=[WrongWord(word="cat", kid_translation="狗", correct_translation="猫")],
    )
    llm = _make_llm(expected)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只狗很大",
    )
    assert result.correct_translation == "那只猫很大"
    assert len(result.wrong_words) == 1
    assert result.wrong_words[0].word == "cat"
    assert result.wrong_words[0].correct_translation == "猫"
    assert result.wrong_words[0].kid_translation == "狗"


@pytest.mark.asyncio
async def test_evaluate_no_wrong_words():
    expected = TranslationEval(correct_translation="那只猫很大", wrong_words=[])
    llm = _make_llm(expected)
    result = await evaluate_translation(
        llm,
        sentence="The cat is big.",
        words=["the", "cat", "is", "big"],
        target="cat",
        kid_input="那只猫很大",
    )
    assert result.wrong_words == []
    assert result.correct_translation == "那只猫很大"


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
    assert result.correct_translation == ""
