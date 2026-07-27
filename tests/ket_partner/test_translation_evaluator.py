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


@pytest.mark.asyncio
async def test_evaluate_threads_target_context_into_prompt():
    """Spec §8.3: target_context shows up in the 'Target word being tested'
    line so the evaluator knows which sense to grade against."""
    expected = TranslationEval(correct_translation="聪明的孩子", wrong_words=[])
    llm = _make_llm(expected)
    await evaluate_translation(
        llm,
        sentence="The smart kid.",
        words=["the", "smart", "kid"],
        target="smart",
        target_context="clever",
        kid_input="聪明的孩子",
    )
    bound = llm.with_structured_output.return_value
    messages = bound.ainvoke.call_args.args[0]
    target_msg = next(m for m in messages if "Target word being tested" in m.content)
    assert "smart" in target_msg.content
    assert "clever" in target_msg.content
