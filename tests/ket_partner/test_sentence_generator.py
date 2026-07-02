from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.sentence_generator import generate_sentence, rewrite_sentence


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
async def test_rewrite_sentence_returns_string():
    llm = _make_llm("The cat sleeps on the warm towel.")
    result = await rewrite_sentence(
        llm,
        original="The cat slept on the warm towel.",
        replace_words=["slept"],
        target="towel",
        age=8,
        min_words=5,
        max_words=12,
    )
    assert isinstance(result, str)
    assert "towel" in result


@pytest.mark.asyncio
async def test_rewrite_sentence_prompt_contains_replace_words():
    """The system prompt MUST surface the failing words so the LLM knows
    what to swap. Without this guarantee the rewrite is blind."""
    llm = _make_llm("The dog runs.")
    await rewrite_sentence(
        llm,
        original="The elephant runs fast.",
        replace_words=["elephant"],
        target="runs",
        age=8,
        min_words=5,
        max_words=12,
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "elephant" in system_text
    assert "runs" in system_text  # target preserved
    assert "The elephant runs fast." in system_text  # original included


@pytest.mark.asyncio
async def test_rewrite_sentence_falls_back_to_original_on_error():
    """If the LLM call throws, return the original sentence unchanged
    rather than crashing the generate_sentence_node retry loop."""
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
    llm.bind = MagicMock(return_value=bound)
    original = "The cat slept on the towel."
    result = await rewrite_sentence(
        llm,
        original=original,
        replace_words=["slept"],
        target="towel",
        age=8,
        min_words=5,
        max_words=12,
    )
    assert result == original


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
async def test_rewrite_sentence_prompt_lists_avoid_sentences():
    """Rewrite path must also receive prior sentences so it doesn't grind
    the sentence back to a previously-used one."""
    llm = _make_llm("A fresh new sentence.")
    await rewrite_sentence(
        llm,
        original="The cat sleeps.",
        replace_words=["sleeps"],
        target="cat",
        age=8,
        min_words=5,
        max_words=12,
        avoid_sentences=["The cat rests."],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "The cat rests." in system_text


@pytest.mark.asyncio
async def test_rewrite_sentence_prompt_does_not_lock_to_original_length():
    """Rewrite must be free to restructure — telling the LLM 'length stays'
    signaled 'match the original's word count' and forced padding with
    non-KET filler. The prompt must explicitly allow restructuring as long
    as the result lands in the configured min-max range.
    """
    llm = _make_llm("A fresh new sentence.")
    await rewrite_sentence(
        llm,
        original="The cat sleeps on the warm bed.",
        replace_words=["sleeps"],
        target="cat",
        age=8,
        min_words=5,
        max_words=12,
    )
    system_text = llm.bind.return_value.ainvoke.call_args.args[0][0].content
    assert "Length stays" not in system_text, (
        "prompt must not say 'Length stays' — that signaled 'match original length'"
    )
    assert "freely restructure" in system_text.lower(), (
        "prompt must explicitly permit restructuring"
    )
