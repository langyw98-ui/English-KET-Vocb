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


@pytest.mark.asyncio
async def test_generate_sentence_prompt_lists_avoid_sentences():
    """The system prompt must surface prior sentences verbatim so the LLM
    knows what NOT to output. Without this, sentence-level dedup is blind."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        avoid_sentences=["The cat runs.", "The cat jumps."],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "The cat runs." in system_text
    assert "The cat jumps." in system_text


@pytest.mark.asyncio
async def test_generate_sentence_prompt_shows_none_yet_when_empty():
    """Empty avoid list must render gracefully (no formatting artifacts)."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        avoid_sentences=[],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "(none yet)" in system_text


@pytest.mark.asyncio
async def test_generate_sentence_prompt_lists_avoid_non_ket_words():
    """Regen path: prior non-KET words must surface in the prompt so the LLM
    knows what to avoid this round. Without this, the LLM keeps producing the
    same non-KET word on every retry."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="build",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        avoid_non_ket_words=["blocks", "tower"],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "blocks" in system_text
    assert "tower" in system_text


@pytest.mark.asyncio
async def test_generate_sentence_prompt_omits_non_ket_block_when_empty():
    """First attempt (no prior non-KET words): block should not render."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        avoid_non_ket_words=[],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "non-KET words" not in system_text
