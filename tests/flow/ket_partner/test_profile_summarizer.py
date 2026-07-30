from unittest.mock import AsyncMock, MagicMock

import pytest

from flow.ket_partner.profile_summarizer import (
    ProfileSummary,
    run_profile_summary,
)
from src.persistence.bootstrap import init_db
from src.persistence.repos import Repos


def _make_llm(return_value):
    llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=return_value)
    llm.with_structured_output = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_summary_updates_profile(temp_db_path):
    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    try:
        await repos.log.append("ai", "The cat is big.", ["cat"], [{"word": "cat", "context": ""}], turn_id=1)
        await repos.log.append("user", "那只狗很大", [], None, turn_id=1)
        llm = _make_llm(ProfileSummary(
            weakness_words=["cat"],
            dialogue_strategy="cat 反复译错,需用图示强化。",
        ))
        await run_profile_summary(llm, repos)
        profile = await repos.profile.get()
        assert "cat" in profile["weakness_words"]
        assert "图示" in profile["dialogue_strategy"]
        llm.with_structured_output.return_value.ainvoke.assert_awaited_once()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_summary_fallback_on_error(temp_db_path):
    db = await init_db(temp_db_path, csv_path=None)
    repos = Repos.for_user(db, "default")
    try:
        await repos.profile.update(weakness_words=["old"], dialogue_strategy="old strat")
        llm = MagicMock()
        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
        llm.with_structured_output = MagicMock(return_value=bound)
        await run_profile_summary(llm, repos)
        profile = await repos.profile.get()
        assert profile["weakness_words"] == ["old"]
        assert profile["dialogue_strategy"] == "old strat"
        bound.ainvoke.assert_awaited_once()
    finally:
        await db.close()

