"""Unit tests for flow.ket_partner.agent.KETPartnerAgent.

Scope: only the _run_summary_safe background-task wrapper. Other node methods
delegate to nodes.py / domain modules and are covered by their own test files.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIError

from flow.ket_partner.agent import _LLM_RETRYABLE, KETPartnerAgent
from flow.ket_partner.config import KetConfig
from flow.ket_partner.dialogue_domain import ProfileSummary

# ===========================================================================
# _LLM_RETRYABLE constant contract
# ===========================================================================

def test_llm_retryable_excludes_code_bug_types():
    """CLAUDE.md §1.5: cross-boundary exception tuples must NOT swallow
    code-bug types (ValueError/TypeError/KeyError/AttributeError/IndexError).
    Those must surface and fail tests rather than be silently absorbed as
    external failures.
    """
    forbidden = {ValueError, TypeError, KeyError, AttributeError, IndexError}
    assert not (set(_LLM_RETRYABLE) & forbidden), (
        f"_LLM_RETRYABLE must not include code-bug types; intersection={set(_LLM_RETRYABLE) & forbidden}"
    )


def test_llm_retryable_includes_expected_external_failures():
    """_LLM_RETRYABLE must cover the three external failure families:
    openai SDK errors, asyncio timeouts, and pydantic schema violations.
    """
    import asyncio

    import openai
    from pydantic import ValidationError

    assert openai.APIError in _LLM_RETRYABLE
    assert asyncio.TimeoutError in _LLM_RETRYABLE
    assert ValidationError in _LLM_RETRYABLE


# ===========================================================================
# _run_summary_safe fallback behavior
# ===========================================================================

def _make_summary_llm(side_effect):
    """Build a mock llm_smart whose structured-output call raises side_effect."""
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=side_effect)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm, bound


def _make_repos() -> MagicMock:
    """Repos mock satisfying the reads run_profile_summary performs."""
    repos = MagicMock()
    repos.profile.get = AsyncMock(
        return_value={"weakness_words": ["old"], "dialogue_strategy": "old strat"}
    )
    repos.profile.update = AsyncMock()
    repos.log.recent = AsyncMock(return_value=[])
    return repos


def _make_llm_service(llm_smart: MagicMock) -> MagicMock:
    """构造满足 LlmService Protocol 的 mock service。

    KETPartnerAgent 通过 LlmService DI 接收 LLM(Phase 2),
    不再直接持有 BaseChatModel。smart/flash 通过属性暴露。
    """
    svc = MagicMock()
    svc.smart = llm_smart
    svc.flash = MagicMock()
    return svc


@pytest.mark.asyncio
async def test_run_summary_safe_swallows_openai_api_error():
    """_run_summary_safe is a background-task wrapper: it MUST NOT raise
    when the underlying LLM call fails with an openai.APIError. Otherwise
    the create_task caller would see an unhandled exception.
    """
    llm_smart, bound = _make_summary_llm(
        APIError(message="llm down", request=MagicMock(), body=None)
    )
    agent = KETPartnerAgent(
        _make_llm_service(llm_smart),
        KetConfig(),
    )
    repos = _make_repos()

    # Must not raise.
    await agent._run_summary_safe(repos)

    # Confirm the LLM path was actually exercised (not silently short-circuited).
    llm_smart.with_structured_output.assert_called_once_with(
        ProfileSummary, method="function_calling"
    )
    bound.ainvoke.assert_awaited_once()
    # Profile must be preserved (not overwritten by a failed summary).
    repos.profile.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_summary_safe_propagates_code_bug():
    """CLAUDE.md §1.5: a code bug (e.g. AttributeError from a typo) must
    NOT be silently swallowed by the retryable tuple. It must surface so
    a test can catch it.
    """
    llm_smart, bound = _make_summary_llm(AttributeError("simulated code bug"))
    agent = KETPartnerAgent(
        _make_llm_service(llm_smart),
        KetConfig(),
    )
    repos = _make_repos()

    with pytest.raises(AttributeError, match="simulated code bug"):
        await agent._run_summary_safe(repos)

    bound.ainvoke.assert_awaited_once()
