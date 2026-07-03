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


@pytest.mark.asyncio
async def test_lookup_retries_when_llm_echoes_english_word():
    """Regression: when the LLM echoes the English word back instead of
    translating (seen in production — kid asked about "sea", got meaning="sea"
    rendered as `「sea」的意思是「sea」`), the function must retry once. If the
    retry produces real Chinese, that's the result."""
    llm = _make_llm(WordMeaning(meaning="海"))
    # First call echoes; second call returns proper Chinese.
    llm.with_structured_output.return_value.ainvoke.side_effect = [
        WordMeaning(meaning="sea"),
        WordMeaning(meaning="海"),
    ]
    result = await lookup_word_meaning(llm, "My dad loves the sea.", "sea")
    assert result.meaning == "海", (
        f"retry after echo must return Chinese; got {result.meaning!r}"
    )
    # Verify the retry actually happened.
    assert llm.with_structured_output.return_value.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_lookup_falls_back_when_both_calls_echo_english():
    """If both attempts echo the English word (no Chinese chars), return the
    explicit failure marker — never the raw English echo."""
    llm = _make_llm(WordMeaning(meaning="sea"))
    llm.with_structured_output.return_value.ainvoke.side_effect = [
        WordMeaning(meaning="sea"),
        WordMeaning(meaning="Sea"),
    ]
    result = await lookup_word_meaning(llm, "My dad loves the sea.", "sea")
    # The fallback marker names the word that failed (so the kid knows which
    # lookup failed) but is wrapped in Chinese context the kid can read.
    assert "失败" in result.meaning, (
        f"persistent English echo must fall back to failure marker; got {result.meaning!r}"
    )
    # The bare English echo must NOT be the entire meaning field — that's the
    # useless `meaning="sea"` we're guarding against.
    assert result.meaning.strip().lower() != "sea", (
        f"meaning must not be the bare English echo; got {result.meaning!r}"
    )
