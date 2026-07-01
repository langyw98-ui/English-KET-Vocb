from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.input_classifier import (
    IntentClassification,
    classify_intent,
)


def _make_llm(return_value: IntentClassification):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_classify_translation():
    expected = IntentClassification(intent="translation", asked_word=None)
    llm = _make_llm(expected)
    result = await classify_intent(llm, "The cat is big.", "那只猫很大")
    assert result.intent == "translation"
    assert result.asked_word is None


@pytest.mark.asyncio
async def test_classify_idk():
    expected = IntentClassification(intent="idk", asked_word=None)
    llm = _make_llm(expected)
    result = await classify_intent(llm, "The cat is big.", "我不会")
    assert result.intent == "idk"


@pytest.mark.asyncio
async def test_classify_asks_meaning_extracts_word():
    expected = IntentClassification(intent="asks_meaning", asked_word="cat")
    llm = _make_llm(expected)
    result = await classify_intent(llm, "The cat is big.", "cat 是什么意思")
    assert result.intent == "asks_meaning"
    assert result.asked_word == "cat"


@pytest.mark.asyncio
async def test_classify_fallback_on_llm_error():
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await classify_intent(llm, "The cat is big.", "那只猫很大")
    assert result.intent == "translation"
