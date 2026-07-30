"""Unit tests for sentence_orchestration extracted from agent.py.

These tests are NEW (the logic was previously inlined as private methods on
KETPartnerAgent without unit-level isolation). All dependencies (LLM, repos,
the underlying validate / generate / naturalness / target-switch functions)
are mocked, so the orchestration control flow can be exercised deterministically.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.sentence_domain import (
    ValidationResult,
    apply_multiword_target_patch,
    generate_with_fallback,
    validate_and_categorize,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_repos() -> MagicMock:
    """KETPartnerRepos is a structural Protocol; a MagicMock satisfies it for
    unit tests because the orchestration code under test never awaits repos.*
    directly (validate_sentence / select_target_word do, and those are patched
    at the module boundary instead)."""
    return MagicMock(spec=[])


def _config(retry_limit: int = 2) -> MagicMock:
    """Minimal config stub: only validate_retry_limit + sentence.min_words /
    sentence.max_words are read by generate_with_fallback."""
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
    """Build a validate_and_categorize return-value dict."""
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


# ---------------------------------------------------------------------------
# validate_and_categorize — 5 paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_and_categorize_passes_clean_sentence(
    monkeypatch: pytest.MonkeyPatch,
):
    """0 non-KET, not duplicate, not target-split, naturalness ok
    -> passed=True, reason_kind=None."""
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
    """Multi-word target whose words do not appear contiguously
    -> reason_kind='target_split'."""
    # 'go shopping' broken apart: 'shopping' present but 'go' missing.
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
    # naturalness must NOT be called when target_split short-circuits the
    # outer branch (the function only calls it when non_ket_count <= 1 and
    # the other guards pass).
    naturalness_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_and_categorize_non_ket_overflow_reason(
    monkeypatch: pytest.MonkeyPatch,
):
    """2+ non-KET words -> reason_kind='non_ket_overflow'."""
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
    """Sentence equals one in avoid_sentences -> reason_kind='duplicate'."""
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
    """0 non-KET, not duplicate, not target-split, but naturalness fails
    -> reason_kind='naturalness'."""
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
    # CLAUDE.md §六.4: also assert the naturalness mock was awaited, otherwise
    # the function could have short-circuited and still produced this reason.
    naturalness_mock.assert_awaited_once()
    validate_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# generate_with_fallback — 3 paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_with_fallback_returns_first_passing(
    monkeypatch: pytest.MonkeyPatch,
):
    """First attempt passes -> returns immediately, no retry loop."""
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
    select_mock = AsyncMock()  # must not be called on the happy path

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
    # Single generate + single validate_orchestration call — no retry happened.
    gen_mock.assert_awaited_once()
    validate_orch_mock.assert_awaited_once()
    # select_target_word must NOT be called on the happy path.
    select_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_with_fallback_switches_target_on_all_naturalness_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    """All attempts fail with 'naturalness' + word_switched=False ->
    calls select_target_word, retries with new target. After the switch the
    next attempt passes."""
    llm = MagicMock()
    repos = _make_repos()
    profile = {"in_refill_mode": 0}
    cfg = _config(retry_limit=2)
    initial_sentence = "The book sings a song."
    switched_sentence = "The dog runs in the park."
    result_initial = ValidationResult(ok=True, words_used=["book"], non_ket_words=[])
    result_switched = ValidationResult(ok=True, words_used=["dog"], non_ket_words=[])

    # retry_limit=2 -> per outer loop: 1 initial + up to 2 inner retries of
    # generate_sentence; validate_and_categorize is called 2 + 1 = 3 times
    # per failed outer iteration. Iteration 1 (initial target) exhausts all
    # retries -> 3 generate calls + 3 fail validates. Iteration 2 (switched
    # target) passes on the first validate -> 1 generate call + 1 pass
    # validate. Total: 4 generate calls + 4 validate calls.
    gen_mock = AsyncMock(
        side_effect=[
            initial_sentence, initial_sentence, initial_sentence,  # initial target
            switched_sentence,                                       # switched target
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
            fail_check, fail_check, fail_check,  # initial target inner+outer
            pass_check,                            # switched target
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
    # select_target_word must have been invoked once to perform the switch.
    select_mock.assert_awaited_once()
    # Four sentence generations: 3 for the initial target (1 + 2 inner
    # retries), then 1 for the switched target (passes on first validate).
    assert gen_mock.await_count == 4
    # Validate the new target was carried into the first switched-target
    # generate call (index 3 = first call after the 3 initial-target calls).
    first_switched_call = gen_mock.await_args_list[3]
    assert first_switched_call.kwargs["target"] == "dog"


@pytest.mark.asyncio
async def test_generate_with_fallback_accepts_overflow_after_retry_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    """All attempts fail with 'non_ket_overflow' -> picks fewest-non-KET draft.
    With retry_limit=2 the loop accumulates overflow attempts; after exhausting
    retries, the function falls into the `overflow_attempts` branch and accepts
    the draft with the smallest non_ket_count, re-validating it once to refresh
    `result`."""
    llm = MagicMock()
    repos = _make_repos()
    profile = {"in_refill_mode": 0}
    cfg = _config(retry_limit=2)
    worst_sentence = "The xylophone ate the zebra."   # 2 non-KET
    best_sentence = "The xylophone ate the apple."    # 1 non-KET

    worst_result = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone", "zebra"]
    )
    best_result = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone"]
    )
    refreshed_result = ValidationResult(
        ok=False, words_used=[], non_ket_words=["xylophone"]
    )

    # retry_limit=2 -> 1 initial + 2 inner retries = 3 generate calls, all
    # producing overflow drafts. The mock cycles worst -> best -> best so
    # the function has both drafts to compare; the function picks the one
    # with the smallest non_ket_count (best_sentence, count=1).
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
    # retry_limit=2: outer sentence + 2 inner retries = 3 fail checks before
    # falling through; the third check is the one that fails after the inner
    # loop. We feed two overflow drafts (worst then best) so the function
    # has both to choose from.
    validate_orch_mock = AsyncMock(
        side_effect=[worst_check, best_check, best_check]
    )
    select_mock = AsyncMock()  # must NOT be called on overflow branch

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

    # Accepted the fewest-non-KET draft.
    assert out[0] == best_sentence
    assert out[1] is refreshed_result
    # select_target_word must NOT have been called on the overflow branch.
    select_mock.assert_not_awaited()
    # validate_sentence (the fresh re-validation of the accepted draft) was
    # called exactly once.
    validate_sentence_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# apply_multiword_target_patch — 1 path
# ---------------------------------------------------------------------------

def test_apply_multiword_target_patch_adds_target_to_words_used():
    """Multi-word target 'ice cream' appears in sentence but not in words_used
    -> patch adds 'ice cream', removes constituent words 'ice' and 'cream'."""
    result = ValidationResult(
        ok=True, words_used=["I", "ice", "cream"], non_ket_words=["ice", "cream"]
    )
    apply_multiword_target_patch("ice cream", "I like ice cream", result)
    assert "ice cream" in result.words_used
    assert "ice" not in result.words_used
    assert "cream" not in result.words_used
    # Constituents must also be scrubbed from non_ket_words.
    assert "ice" not in result.non_ket_words
    assert "cream" not in result.non_ket_words
    # The non-target word 'I' must survive untouched.
    assert "I" in result.words_used
