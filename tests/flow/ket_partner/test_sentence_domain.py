import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.sentence_domain import (
    _NATURALNESS_SYSTEM as _SYSTEM,
)
from flow.ket_partner.sentence_domain import (
    NaturalnessResult,
    ValidationResult,
    apply_multiword_target_patch,
    build_target_pattern,
    check_naturalness,
    find_placeholder,
    generate_sentence,
    generate_with_fallback,
    has_placeholder,
    target_in_sentence,
    validate_and_categorize,
    validate_sentence,
)
from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos

# ===========================================================================
# Generator Tests
# ===========================================================================

def _make_generator_llm(return_value: str):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=MagicMock(content=return_value))
    llm.bind = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_generate_sentence_returns_string():
    llm = _make_generator_llm("The big cat is sleeping on the bed.")
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
    llm = _make_generator_llm("I see a cat.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    llm = _make_generator_llm("A fresh new sentence.")
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
    assert "Let us build a tall tower with blocks." in system_text
    assert "We build a house of bricks today." in system_text
    assert "non-KET words" in system_text
    assert "house of bricks sounds odd" in system_text
    bound.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_sentence_prompt_clarifies_multi_word_target():
    """When target is a multi-word phrase, the prompt must call out that its
    words are inseparable. Without this the LLM splits them ('CD player' →
    '...CD into the old player.')."""
    llm = _make_generator_llm("She has a new CD player.")
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
    llm = _make_generator_llm("The cat runs.")
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
    llm = _make_generator_llm("She has a new CD player.")
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
    llm = _make_generator_llm("The cat runs.")
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
    llm = _make_generator_llm("The smart kid solved the puzzle.")
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
    llm = _make_generator_llm("The cat runs.")
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


@pytest.mark.asyncio
async def test_generate_sentence_falls_back_on_llm_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回含 target 的模板句,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="llm down",
            request=MagicMock(),
            body=None,
        )
    )
    llm.bind = MagicMock(return_value=bound)
    result = await generate_sentence(
        llm,
        target="cat",
        recent_scaffolding=[],
        age=8,
        min_words=5,
        max_words=12,
    )
    assert isinstance(result, str)
    assert "cat" in result
    bound.ainvoke.assert_awaited_once()


# ===========================================================================
# Validator Tests
# ===========================================================================

@pytest.fixture
async def repos(temp_db_path):
    csv_text = (
        "word,part_of_speech,topic,context\n"
        "cat,n,Animals,\n"
        "dog,n,Animals,\n"
        "big,adj,,\n"
        "small,adj,,\n"
        "happy,adj,,\n"
        "is,v,,\n"
        "am,v,,\n"
        "are,v,,\n"
        "go,v,Action,\n"
        "I,pron,,\n"
        "the,det,,\n"
        "a,det,,\n"
        "on,prep,,\n"
        "bed,n,,\n"
        "hat,n,Clothing,\n"
        "cake,n,Food,\n"
        "wear,v,Clothing,\n"
        "say,v,Action,\n"
        "and,conj,,\n"
        "watch,v,Action,\n"
        "walk,v,Action,\n"
        "run,v,Action,\n"
        "bake,v,Food,\n"
        "bake,v,Food,\n"
        "story,n,Reading,\n"
        "try,v,Action,\n"
        "large,adj,Size,\n"
        "make,v,Action,\n"
        "rain,n,Weather,\n"
        "grass,n,Nature,\n"
        "wet,adj,,\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_text)
        csv_path = f.name
    db = await init_db(temp_db_path, csv_path=csv_path)
    r = Repos.for_user(db, "default")
    yield r
    await db.close()


@pytest.mark.asyncio
async def test_validate_all_ket_words_passes(repos):
    result = await validate_sentence("The cat is on the bed.", repos)
    assert result.ok is True
    assert "cat" in result.words_used


@pytest.mark.asyncio
async def test_validate_non_ket_word_fails(repos):
    result = await validate_sentence("The cat is on the elephant.", repos)
    assert result.ok is False
    assert "elephant" in result.non_ket_words


@pytest.mark.asyncio
async def test_validate_lemma_reduces_to_ket_root(repos):
    result = await validate_sentence("The cats are on the bed.", repos)
    assert result.ok is True


@pytest.mark.asyncio
async def test_validate_handles_verb_inflections(repos):
    r1 = await validate_sentence("The cat wears a hat.", repos)
    assert r1.ok is True, f"wears should reduce to wear; got non_ket={r1.non_ket_words}"
    assert "wear" in r1.words_used
    r2 = await validate_sentence("The dog walked.", repos)
    assert r2.ok is True
    assert "walk" in r2.words_used
    r3 = await validate_sentence("The big cake baked.", repos)
    assert r3.ok is True, f"baked should reduce to bake; got non_ket={r3.non_ket_words}"
    assert "bake" in r3.words_used
    r4 = await validate_sentence("The dog is running.", repos)
    assert r4.ok is True, f"running should reduce to run; got non_ket={r4.non_ket_words}"
    assert "run" in r4.words_used
    r5 = await validate_sentence("The big cake is baking.", repos)
    assert r5.ok is True, f"baking should reduce to bake; got non_ket={r5.non_ket_words}"
    assert "bake" in r5.words_used


@pytest.mark.asyncio
async def test_validate_handles_noun_plurals(repos):
    r1 = await validate_sentence("The happy stories.", repos)
    assert r1.ok is True, f"stories should reduce to story; got non_ket={r1.non_ket_words}"
    assert "story" in r1.words_used
    r2 = await validate_sentence("The dog watches.", repos)
    assert r2.ok is True, f"watches should reduce to watch; got non_ket={r2.non_ket_words}"
    assert "watch" in r2.words_used


@pytest.mark.asyncio
async def test_validate_handles_comparatives_and_superlatives(repos):
    r1 = await validate_sentence("The happier dog walked.", repos)
    assert r1.ok is True, f"happier should reduce to happy; got non_ket={r1.non_ket_words}"
    assert "happy" in r1.words_used
    r2 = await validate_sentence("The bigger dog walked.", repos)
    assert r2.ok is True
    assert "big" in r2.words_used
    r3 = await validate_sentence("The larger dog walked.", repos)
    assert r3.ok is True, f"larger should reduce to large; got non_ket={r3.non_ket_words}"
    assert "large" in r3.words_used
    r4 = await validate_sentence("The happiest dog walked.", repos)
    assert r4.ok is True
    assert "happy" in r4.words_used
    r5 = await validate_sentence("The biggest dog walked.", repos)
    assert r5.ok is True
    assert "big" in r5.words_used


@pytest.mark.asyncio
async def test_validate_verb_s_form_after_e_verb(repos):
    r = await validate_sentence("The rain makes the grass wet.", repos)
    assert r.ok is True, f"makes should reduce to make; got non_ket={r.non_ket_words}"
    assert "make" in r.words_used


@pytest.mark.asyncio
async def test_validate_lemma_overrides_rule_based(repos):
    r = await validate_sentence("The dog went big.", repos)
    assert "go" in r.words_used, "irregular lemma 'went→go' must take precedence"


@pytest.mark.asyncio
async def test_validate_pronoun_I_case_insensitive(repos):
    r = await validate_sentence("I am big.", repos)
    assert r.ok is True, f"'I' must match the vocab row stored as 'I'; got non_ket={r.non_ket_words}"
    assert "I" in r.words_used, "canonical form 'I' must be recorded, not lowercase 'i'"


@pytest.mark.asyncio
async def test_validator_recognizes_word_with_only_specific_contexts_as_ket(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('smart', 'clever', 'adj', 0), ('smart', 'stylish', 'adj', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The smart dog runs.", repos)
    assert r.ok is True, f"smart must be KET; got non_ket={r.non_ket_words}"
    assert "smart" in r.words_used


@pytest.mark.asyncio
async def test_validate_verb_uses_lemmatizes_to_use_not_us(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('use', '', 'v', 0), ('us', '', 'pron', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The cat uses the bed.", repos)
    assert r.ok is True, f"uses should reduce to use; got non_ket={r.non_ket_words}"
    assert "use" in r.words_used, f"uses should reduce to 'use' (verb), got {r.words_used}"
    assert "us" not in r.words_used, "pronoun 'us' must not steal the verb's stats"


@pytest.mark.asyncio
async def test_validator_recognizes_multi_word_target_as_ket_unit(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('alarm clock', '', 'n', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence(
        "The alarm clock is on the bed.",
        repos,
        target="alarm clock",
    )
    assert r.ok is True, f"alarm clock should be KET as a unit; got non_ket={r.non_ket_words}"
    assert "alarm clock" in r.words_used
    assert "alarm" not in r.words_used, "constituent must not double-count as scaffolding"


@pytest.mark.asyncio
async def test_validator_recognizes_placeholder_target_as_ket_unit(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('give somebody a call', '', 'v', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence(
        "I give the dog a call.",
        repos,
        target="give somebody a call",
    )
    assert r.ok is True, f"placeholder target should be KET as a unit; got non_ket={r.non_ket_words}"
    assert "give somebody a call" in r.words_used
    assert "give" not in r.words_used, "constituent 'give' must not double-count"
    assert "call" not in r.words_used, "constituent 'call' must not double-count"


@pytest.mark.asyncio
async def test_validator_recognizes_hyphenated_target_as_single_token(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('guest-house', '', 'n', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence(
        "The guest-house is big.",
        repos,
        target="guest-house",
    )
    assert r.ok is True, f"hyphenated target should be KET; got non_ket={r.non_ket_words}"
    assert "guest-house" in r.words_used
    assert "guest" not in r.words_used
    assert "house" not in r.words_used


@pytest.mark.asyncio
async def test_validator_recognizes_capitalized_hyphenated_word_in_middle(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('T-shirt', '', 'n', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The T-shirt is big.", repos)
    assert r.ok is True, f"T-shirt in middle position should be KET; got non_ket={r.non_ket_words}"
    assert "T-shirt" in r.words_used, "T-shirt must be recorded in canonical form"


@pytest.mark.asyncio
async def test_validator_recognizes_acronym_in_middle_position(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('DVD', '', 'n', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The DVD is big.", repos)
    assert r.ok is True, f"DVD in middle position should be KET; got non_ket={r.non_ket_words}"
    assert "DVD" in r.words_used, "DVD must be recorded in canonical form"


@pytest.mark.asyncio
async def test_validator_still_skips_unknown_proper_nouns_in_middle(repos):
    r = await validate_sentence("The cat makes John happy.", repos)
    assert r.ok is True, f"unknown proper noun 'John' should be tolerated; got non_ket={r.non_ket_words}"
    assert "John" not in r.words_used, "John is not KET — must not appear in words_used"
    assert "John" not in r.non_ket_words, "John at i>0 must be skipped, not flagged"


@pytest.mark.asyncio
async def test_validator_recognizes_exclamation_entry_stripped_of_punctuation(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('congratulations!', '', 'exclam', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The cat says congratulations.", repos)
    assert r.ok is True, f"exclamation entry should match stripped form; got non_ket={r.non_ket_words}"
    assert "congratulations!" in r.words_used, "canonical form 'congratulations!' must be recorded"


@pytest.mark.asyncio
async def test_validator_recognizes_capitalized_exclamation_in_middle(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('Yeah!', '', 'exclam', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The cat says Yeah! and runs.", repos)
    assert r.ok is True, f"'Yeah!' should be KET; got non_ket={r.non_ket_words}"
    assert "Yeah!" in r.words_used, "canonical form 'Yeah!' must be recorded"


@pytest.mark.asyncio
async def test_validator_recognizes_abbreviation_with_periods(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('p.m.', '', 'adv', 0), ('a.m.', '', 'adv', 0)"
    )
    await repos.vocab._db.commit()
    r1 = await validate_sentence("The p.m. is big.", repos, target="p.m.")
    assert r1.ok is True, f"p.m. should be KET; got non_ket={r1.non_ket_words}"
    assert "p.m." in r1.words_used

    r2 = await validate_sentence("The a.m. is big.", repos, target="a.m.")
    assert r2.ok is True, f"a.m. should be KET; got non_ket={r2.non_ket_words}"
    assert "a.m." in r2.words_used


@pytest.mark.asyncio
async def test_validator_strips_terminal_punctuation_from_target_constituents(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('Guess what?', '', 'v', 0), ('what', '', 'pron', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("Guess what? The cat is big.", repos, target="Guess what?")
    assert r.ok is True
    assert "Guess what?" in r.words_used, "phrase must be recorded"
    assert "what" not in r.words_used, "constituent 'what' must not double-count"


@pytest.mark.asyncio
async def test_validator_handles_trailing_period_in_sentence_final_word(repos):
    r = await validate_sentence("The cat is big.", repos)
    assert r.ok is True, f"trailing period must not break lookup; got non_ket={r.non_ket_words}"
    assert "big" in r.words_used


@pytest.mark.asyncio
async def test_validator_picks_lowercase_for_sentence_initial_capitalized_token(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('it', '', 'pron', 0), ('IT', '', 'n', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("It is a cat.", repos)
    assert r.ok is True
    assert "it" in r.words_used, (
        f"capitalized 'It' (first-letter-only) must map to pronoun 'it', "
        f"got {r.words_used}"
    )
    assert "IT" not in r.words_used


@pytest.mark.asyncio
async def test_validator_picks_uppercase_for_allcaps_token_in_middle(repos):
    await repos.vocab._db.execute(
        "INSERT INTO ket_vocabulary (word, context, pos, is_seed) VALUES "
        "('it', '', 'pron', 0), ('IT', '', 'n', 0)"
    )
    await repos.vocab._db.commit()
    r = await validate_sentence("The cat and IT are big.", repos)
    assert r.ok is True
    assert "IT" in r.words_used, (
        f"all-caps 'IT' must map to abbreviation 'IT', got {r.words_used}"
    )
    assert "it" not in r.words_used


# ===========================================================================
# Naturalness Tests
# ===========================================================================

def _make_naturalness_llm(return_value: NaturalnessResult):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_check_naturalness_returns_ok():
    llm = _make_naturalness_llm(NaturalnessResult(ok=True, reason=""))
    result = await check_naturalness(llm, "The cat sleeps on the bed.")
    assert result.ok is True
    assert result.reason == ""
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_naturalness_returns_reject_with_reason():
    llm = _make_naturalness_llm(NaturalnessResult(ok=False, reason="ice cream does not make noses move"))
    result = await check_naturalness(llm, "The cold ice cream makes my nose move.")
    assert result.ok is False
    assert "nose" in result.reason
    llm.with_structured_output.return_value.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_naturalness_fails_open_on_error():
    """LLM 调用抛 openai.APIError 时,fallback 返回 ok=True,且 mock 确实被调用过。"""
    from openai import APIError

    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=APIError(
            message="llm down",
            request=MagicMock(),
            body=None,
        )
    )
    llm.with_structured_output = MagicMock(return_value=bound)
    result = await check_naturalness(llm, "anything")
    assert result.ok is True
    assert result.reason == ""
    llm.with_structured_output.assert_called_once_with(
        NaturalnessResult, method="function_calling"
    )
    bound.ainvoke.assert_awaited_once()


def test_prompt_covers_three_naturalness_categories():
    assert "ice cream does not make noses move" in _SYSTEM, "category 1 (impossibility) example must be present"
    assert "wet water" in _SYSTEM, "category 2 (redundancy) example must be present"
    assert "Collocation" in _SYSTEM or "collocation" in _SYSTEM, "category 3 (collocation) must be named"


# ===========================================================================
# Orchestration Tests
# ===========================================================================

def _make_repos() -> MagicMock:
    return MagicMock(spec=[])


def _config(retry_limit: int = 2) -> MagicMock:
    cfg = MagicMock()
    cfg.validate_retry_limit = retry_limit
    cfg.sentence.min_words = 5
    cfg.sentence.max_words = 12
    return cfg


def _check(
    result: ValidationResult,
    passed: bool,
    reason_kind: str | None,
    reason_detail: str,
    non_ket_words: list[str],
    non_ket_count: int,
    sentence: str,
    is_duplicate: bool = False,
    is_target_split: bool = False,
) -> dict:
    return {
        "result": result,
        "passed": passed,
        "reason_kind": reason_kind,
        "reason_detail": reason_detail,
        "non_ket_words": non_ket_words,
        "non_ket_count": non_ket_count,
        "is_duplicate": is_duplicate,
        "is_target_split": is_target_split,
        "sentence": sentence,
    }


@pytest.mark.asyncio
async def test_validate_and_categorize_passes_clean_sentence(
    monkeypatch: pytest.MonkeyPatch,
):
    sentence = "The cat sleeps on the bed."
    target = "cat"
    repos = _make_repos()
    llm = MagicMock()

    clean = ValidationResult(ok=True, words_used=["cat"], non_ket_words=[])
    validate_mock = AsyncMock(return_value=clean)
    naturalness_mock = AsyncMock(return_value=MagicMock(ok=True, reason=""))

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_sentence", validate_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.check_naturalness", naturalness_mock
    )

    result = await validate_and_categorize(
        llm, sentence, target, age=8, repos=repos, avoid_sentences=[]
    )

    assert result["passed"] is True
    assert result["reason_kind"] is None
    assert result["reason_detail"] == ""
    assert result["non_ket_count"] == 0
    assert result["is_duplicate"] is False
    assert result["is_target_split"] is False
    validate_mock.assert_awaited_once()
    naturalness_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_validate_and_categorize_target_split_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    sentence = "She went shopping for fruit."
    target = "go shopping"
    repos = _make_repos()
    llm = MagicMock()

    clean = ValidationResult(ok=True, words_used=["go", "shopping"], non_ket_words=[])
    validate_mock = AsyncMock(return_value=clean)
    naturalness_mock = AsyncMock(return_value=MagicMock(ok=True))

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_sentence", validate_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.check_naturalness", naturalness_mock
    )

    result = await validate_and_categorize(
        llm, sentence, target, age=8, repos=repos, avoid_sentences=[]
    )

    assert result["passed"] is False
    assert result["reason_kind"] == "target_split"
    assert "multi-word target 'go shopping'" in result["reason_detail"]
    assert result["is_target_split"] is True
    naturalness_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_and_categorize_non_ket_overflow_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    sentence = "The xylophone ate the zebra."
    target = ""
    repos = _make_repos()
    llm = MagicMock()

    overflow = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone", "zebra"]
    )
    validate_mock = AsyncMock(return_value=overflow)
    naturalness_mock = AsyncMock(return_value=MagicMock(ok=True))

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_sentence", validate_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.check_naturalness", naturalness_mock
    )

    result = await validate_and_categorize(
        llm, sentence, target, age=8, repos=repos, avoid_sentences=[]
    )

    assert result["passed"] is False
    assert result["reason_kind"] == "non_ket_overflow"
    assert result["non_ket_count"] == 2
    assert "xylophone" in result["reason_detail"]
    assert "zebra" in result["reason_detail"]
    naturalness_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_and_categorize_duplicate_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    sentence = "The cat sleeps on the bed."
    target = "cat"
    repos = _make_repos()
    llm = MagicMock()

    clean = ValidationResult(ok=True, words_used=["cat"], non_ket_words=[])
    validate_mock = AsyncMock(return_value=clean)
    naturalness_mock = AsyncMock(return_value=MagicMock(ok=True))

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_sentence", validate_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.check_naturalness", naturalness_mock
    )

    result = await validate_and_categorize(
        llm,
        sentence,
        target,
        age=8,
        repos=repos,
        avoid_sentences=[sentence],
    )

    assert result["passed"] is False
    assert result["reason_kind"] == "duplicate"
    assert result["is_duplicate"] is True
    naturalness_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_and_categorize_naturalness_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    sentence = "The book sings a loud song."
    target = "book"
    repos = _make_repos()
    llm = MagicMock()

    clean = ValidationResult(ok=True, words_used=["book"], non_ket_words=[])
    unnatural = MagicMock(ok=False, reason="books do not sing")

    validate_mock = AsyncMock(return_value=clean)
    naturalness_mock = AsyncMock(return_value=unnatural)

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_sentence", validate_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.check_naturalness", naturalness_mock
    )

    result = await validate_and_categorize(
        llm, sentence, target, age=8, repos=repos, avoid_sentences=[]
    )

    assert result["passed"] is False
    assert result["reason_kind"] == "naturalness"
    assert "books do not sing" in result["reason_detail"]
    naturalness_mock.assert_awaited_once()
    validate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_with_fallback_returns_first_passing(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = MagicMock()
    repos = _make_repos()
    profile = {"in_refill_mode": 0}
    cfg = _config(retry_limit=2)
    sentence = "I see a cat."
    result_obj = ValidationResult(ok=True, words_used=["cat"], non_ket_words=[])

    gen_mock = AsyncMock(return_value=sentence)
    validate_orch_mock = AsyncMock(
        return_value=_check(
            result_obj,
            passed=True,
            reason_kind=None,
            reason_detail="",
            non_ket_words=[],
            non_ket_count=0,
            sentence=sentence,
        )
    )
    select_mock = AsyncMock()

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.generate_sentence", gen_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_and_categorize",
        validate_orch_mock,
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.select_target_word", select_mock
    )

    out = await generate_with_fallback(
        llm,
        initial_target="cat",
        initial_context="",
        avoid_words=[],
        avoid_sentences=[],
        age=8,
        profile=profile,
        repos=repos,
        config=cfg,
    )

    assert out[0] == sentence
    assert out[1] is result_obj
    assert out[2] == "cat"
    assert out[3] == ""
    gen_mock.assert_awaited_once()
    validate_orch_mock.assert_awaited_once()
    select_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_with_fallback_switches_target_on_all_naturalness_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = MagicMock()
    repos = _make_repos()
    profile = {"in_refill_mode": 0}
    cfg = _config(retry_limit=2)
    initial_sentence = "The book sings a song."
    switched_sentence = "The dog runs in the park."
    result_initial = ValidationResult(ok=True, words_used=["book"], non_ket_words=[])
    result_switched = ValidationResult(ok=True, words_used=["dog"], non_ket_words=[])

    gen_mock = AsyncMock(
        side_effect=[
            initial_sentence, initial_sentence, initial_sentence,
            switched_sentence,
        ]
    )
    fail_check = _check(
        result_initial,
        passed=False,
        reason_kind="naturalness",
        reason_detail="unnatural expression — books do not sing",
        non_ket_words=[],
        non_ket_count=0,
        sentence=initial_sentence,
    )
    pass_check = _check(
        result_switched,
        passed=True,
        reason_kind=None,
        reason_detail="",
        non_ket_words=[],
        non_ket_count=0,
        sentence=switched_sentence,
    )
    validate_orch_mock = AsyncMock(
        side_effect=[
            fail_check, fail_check, fail_check,
            pass_check,
        ]
    )
    new_ref = MagicMock(word="dog", context="")
    select_mock = AsyncMock(return_value=new_ref)

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.generate_sentence", gen_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_and_categorize",
        validate_orch_mock,
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.select_target_word", select_mock
    )

    out = await generate_with_fallback(
        llm,
        initial_target="book",
        initial_context="",
        avoid_words=[],
        avoid_sentences=[],
        age=8,
        profile=profile,
        repos=repos,
        config=cfg,
    )

    assert out[0] == switched_sentence
    assert out[2] == "dog"
    select_mock.assert_awaited_once()
    assert gen_mock.await_count == 4
    first_switched_call = gen_mock.await_args_list[3]
    assert first_switched_call.kwargs["target"] == "dog"


@pytest.mark.asyncio
async def test_generate_with_fallback_accepts_overflow_after_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    llm = MagicMock()
    repos = _make_repos()
    profile = {"in_refill_mode": 0}
    cfg = _config(retry_limit=2)
    worst_sentence = "The xylophone ate the zebra."
    best_sentence = "The xylophone ate the apple."

    worst_result = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone", "zebra"]
    )
    best_result = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone"]
    )
    refreshed_result = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone"]
    )

    gen_mock = AsyncMock(side_effect=[worst_sentence, best_sentence, best_sentence])
    worst_check = _check(
        worst_result,
        passed=False,
        reason_kind="non_ket_overflow",
        reason_detail="non-KET words ['xylophone', 'zebra'] exceed the limit",
        non_ket_words=["xylophone", "zebra"],
        non_ket_count=2,
        sentence=worst_sentence,
    )
    best_check = _check(
        best_result,
        passed=False,
        reason_kind="non_ket_overflow",
        reason_detail="non-KET words ['xylophone'] exceed the limit",
        non_ket_words=["xylophone"],
        non_ket_count=1,
        sentence=best_sentence,
    )
    validate_orch_mock = AsyncMock(
        side_effect=[worst_check, best_check, best_check]
    )
    select_mock = AsyncMock()
    validate_sentence_mock = AsyncMock(return_value=refreshed_result)

    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.generate_sentence", gen_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_and_categorize",
        validate_orch_mock,
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.select_target_word", select_mock
    )
    monkeypatch.setattr(
        "flow.ket_partner.sentence_domain.validate_sentence",
        validate_sentence_mock,
    )

    out = await generate_with_fallback(
        llm,
        initial_target="apple",
        initial_context="",
        avoid_words=[],
        avoid_sentences=[],
        age=8,
        profile=profile,
        repos=repos,
        config=cfg,
    )

    assert out[0] == best_sentence
    assert out[1] is refreshed_result
    select_mock.assert_not_awaited()
    validate_sentence_mock.assert_awaited_once()


def test_apply_multiword_target_patch_adds_target_to_words_used():
    result = ValidationResult(
        ok=True, words_used=["I", "ice", "cream"], non_ket_words=["ice", "cream"]
    )
    apply_multiword_target_patch("ice cream", "I like ice cream", result)
    assert "ice cream" in result.words_used
    assert "ice" not in result.words_used
    assert "cream" not in result.words_used
    assert "ice" not in result.non_ket_words
    assert "cream" not in result.non_ket_words
    assert "I" in result.words_used


# ===========================================================================
# Multi-Word Target Pattern Tests
# ===========================================================================

@pytest.mark.parametrize("target,expected", [
    ("give somebody a call", True),
    ("give somebody a ring", True),
    ("tell someone a story", True),
    ("buy something", True),
    ("alarm clock", False),
    ("CD player", False),
    ("cat", False),
    ("", False),
    (None, False),
])
def test_has_placeholder(target, expected):
    assert has_placeholder(target) is expected


@pytest.mark.parametrize("sentence,should_match", [
    ("I give my mom a call every night.", True),
    ("He gives him a call.", True),
    ("She is giving the teacher a call.", True),
    ("She gives the tall man a call.", True),
    ("They give the very tall man a call.", False),
    ("I give a call.", False),
    ("I give the dog a walk.", False),
    ("I gave my mom a call.", False),
    ("Please call my mom.", False),
])
def test_build_target_pattern_placeholder_phrase(sentence, should_match):
    pat = build_target_pattern("give somebody a call")
    assert bool(pat.search(sentence)) is should_match


@pytest.mark.parametrize("sentence,should_match", [
    ("The alarm clock rings.", True),
    ("The Alarm Clock rings.", True),
    ("I set the alarm clock.", True),
    ("The clock alarms me.", False),
    ("alarm clocks ring.", True),
])
def test_build_target_pattern_literal_phrase(sentence, should_match):
    pat = build_target_pattern("alarm clock")
    assert bool(pat.search(sentence)) is should_match


@pytest.mark.parametrize("target,sentence,expected", [
    ("alarm clock", "the alarm clock rings", True),
    ("alarm clock", "the clock alarms", False),
    ("give somebody a call", "I give him a call", True),
    ("give somebody a call", "I give a call", False),
    ("cat", "the cat sleeps", True),
])
def test_target_in_sentence(target, sentence, expected):
    assert target_in_sentence(target, sentence) is expected


def test_find_placeholder_returns_first_match():
    assert find_placeholder("give somebody a call") == "somebody"
    assert find_placeholder("tell someone something") == "someone"
    assert find_placeholder("alarm clock") == ""
    assert find_placeholder("") == ""
    assert find_placeholder(None) == ""
