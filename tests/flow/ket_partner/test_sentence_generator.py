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
    llm.bind.return_value.ainvoke.assert_awaited_once()


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
    llm.bind.return_value.ainvoke.assert_awaited_once()


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
    bound.ainvoke.assert_awaited_once()


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
    bound.ainvoke.assert_awaited_once()


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
    bound.ainvoke.assert_awaited_once()


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
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_lists_prior_duplicate_attempt():
    """Regen after an exact-match duplicate: the offending sentence must
    surface in the prior-attempts history so the LLM understands which exact
    wording to avoid — not just the soft avoid_sentences list."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        prior_attempts=[
            {
                "sentence": "The cat runs fast.",
                "reason_kind": "duplicate",
                "reason_detail": "word-for-word duplicate of a recent sentence",
            }
        ],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "The cat runs fast." in system_text
    assert "word-for-word duplicate" in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_omits_history_when_no_prior_attempts():
    """First attempt (no prior failures): history block should not render."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "Your previous attempts" not in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_lists_all_prior_attempts():
    """Every prior failed attempt (sentence + reason) must appear in the
    prompt — not just the latest. This is the core change: the LLM sees the
    full failure history so it can avoid repeating any of them."""
    llm = _make_llm("A fresh new sentence.")
    await generate_sentence(
        llm,
        target="build",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        prior_attempts=[
            {
                "sentence": "Let us build a tall tower with blocks.",
                "reason_kind": "non_ket_overflow",
                "reason_detail": "non-KET words ['tower', 'blocks'] exceed the limit (max 1 allowed)",
            },
            {
                "sentence": "We build a house of bricks today.",
                "reason_kind": "naturalness",
                "reason_detail": "unnatural expression — house of bricks sounds odd",
            },
        ],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    # Both sentences must appear verbatim.
    assert "Let us build a tall tower with blocks." in system_text
    assert "We build a house of bricks today." in system_text
    # Both reasons must appear.
    assert "non-KET words" in system_text
    assert "house of bricks sounds odd" in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_clarifies_multi_word_target():
    """When target is a multi-word phrase, the prompt must call out that its
    words are inseparable. Without this the LLM splits them ('CD player' →
    '...CD into the old player.')."""
    llm = _make_llm("She has a new CD player.")
    await generate_sentence(
        llm,
        target="CD player",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "MULTI-WORD" in system_text
    assert "CD player" in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_omits_multi_word_note_for_single_word():
    """Single-word targets must not render the multi-word note."""
    llm = _make_llm("The cat runs.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "MULTI-WORD" not in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_calls_out_target_split_when_prior_attempt_split():
    """Regen after the LLM split a multi-word target: the prompt must surface
    the failure as a dedicated callout so the LLM understands the structural
    requirement it just violated. Fires when ANY prior attempt was a split."""
    llm = _make_llm("She has a new CD player.")
    await generate_sentence(
        llm,
        target="CD player",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        prior_attempts=[
            {
                "sentence": "He puts a CD into the old player.",
                "reason_kind": "target_split",
                "reason_detail": "split the multi-word target 'CD player' — words must be contiguous",
            }
        ],
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "SPLIT" in system_text
    assert "CD player" in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_omits_target_split_block_when_no_split_history():
    """No prior target_split attempt: block must not render."""
    llm = _make_llm("The cat runs.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "SPLIT" not in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_includes_context_when_provided():
    """Spec §8.1: when target_context is non-empty, the prompt must tell
    the LLM which sense of the target to use. Without this, 'smart' might
    come back as 'stylish' when the system intended 'clever'."""
    llm = _make_llm("The smart kid solved the puzzle.")
    await generate_sentence(
        llm,
        target="smart",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
        target_context="clever",
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "smart" in system_text
    assert "clever" in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_omits_context_block_when_empty():
    """First-turn / single-sense case: no context block."""
    llm = _make_llm("The cat runs.")
    await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    bound = llm.bind.return_value
    sent_messages = bound.ainvoke.call_args.args[0]
    system_text = sent_messages[0].content
    assert "specifically in this sense" not in system_text
    bound.ainvoke.assert_awaited_once()

