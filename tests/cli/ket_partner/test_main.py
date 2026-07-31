"""Tests for cli.ket_partner.main — the CLI composition root.

Both tests drive main() end-to-end with heavy monkeypatching so the real
network, LLM, file IO and DB layers never run. They focus on wiring: that
main() plumbs init_db -> Repos -> build_agent -> CommandHandler together,
and that build_agent is called without a db arg per Task 13's signature.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.cli.ket_partner.commands import ExitLoop
from src.cli.ket_partner.main import main


def _install_wiring(monkeypatch, fake_build: AsyncMock) -> None:
    """Wire every cross-boundary name in main() to a fake.

    Shared by both tests; each test then asserts what it cares about.
    """
    # init_db: async; returns a db connection whose .close() is awaitable.
    fake_db = MagicMock(name="db")
    fake_db.close = AsyncMock(name="db.close")
    fake_init = AsyncMock(name="init_db", return_value=fake_db)
    monkeypatch.setattr("src.cli.ket_partner.main.init_db", fake_init)

    # Repos class: MagicMock; for_user returns a repos whose log.append_session_start is awaitable.
    fake_repos = MagicMock(name="repos")
    fake_repos.log.append_session_start = AsyncMock(name="repos.log.append_session_start")
    fake_repos_class = MagicMock(name="Repos")
    fake_repos_class.for_user.return_value = fake_repos
    monkeypatch.setattr("src.cli.ket_partner.main.Repos", fake_repos_class)

    # build_agent: async; returns an agent whose .agent.aclose() is awaitable so the
    # finally block exercises the shutdown branch without raising.
    fake_agent = MagicMock(name="agent")
    inner = MagicMock(name="agent.agent")
    inner.aclose = AsyncMock(name="agent.agent.aclose")
    fake_agent.agent = inner
    fake_build.return_value = fake_agent
    monkeypatch.setattr("src.cli.ket_partner.main.build_agent", fake_build)

    # ChatLogger: sync class; instance methods are sync. Now used as a context
    # manager (`with ChatLogger(...) as chat_logger`), so __enter__ must return
    # the same fake instance for the assertions to see it.
    fake_chat_logger = MagicMock(name="chat_logger")
    fake_chat_logger.__enter__ = MagicMock(return_value=fake_chat_logger)
    fake_chat_logger.__exit__ = MagicMock(return_value=None)
    fake_chat_logger_class = MagicMock(name="ChatLogger", return_value=fake_chat_logger)
    monkeypatch.setattr("src.cli.ket_partner.main.ChatLogger", fake_chat_logger_class)

    # CommandHandler: sync class; handle() raises ExitLoop on /quit so the loop breaks.
    fake_cmd_handler = MagicMock(name="cmd_handler")
    fake_cmd_handler.handle = AsyncMock(name="cmd_handler.handle", side_effect=ExitLoop())
    fake_cmd_handler_class = MagicMock(name="CommandHandler", return_value=fake_cmd_handler)
    monkeypatch.setattr("src.cli.ket_partner.main.CommandHandler", fake_cmd_handler_class)

    # First prompt returns /quit immediately; asyncio.to_thread(input, ...) picks this up.
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "/quit")


@pytest.mark.asyncio
async def test_main_initializes_and_loops(monkeypatch):
    """main() wires init_db + Repos + build_agent + CommandHandler and exits
    cleanly on /quit. Verifies the wiring without exercising the real input
    loop or any cross-boundary call."""
    fake_build = AsyncMock(name="build_agent")
    _install_wiring(monkeypatch, fake_build)

    # Should return without raising; ExitLoop is caught internally and breaks the loop.
    await main()

    # Sanity: every wired boundary was actually reached. If main() had short-circuited
    # via an exception these would all be zero, so asserting call counts guards against
    # silent fallback paths (CLAUDE.md §六.4).
    from src.cli.ket_partner import main as main_mod

    assert main_mod.init_db.await_count == 1
    assert main_mod.Repos.for_user.call_count == 1
    assert main_mod.Repos.for_user.return_value.log.append_session_start.await_count == 1
    assert main_mod.build_agent.await_count == 1
    assert main_mod.ChatLogger.call_count == 1
    assert main_mod.CommandHandler.call_count == 1
    # /quit was handled, and that handle raised ExitLoop to break the loop.
    cmd_handler_instance = main_mod.CommandHandler.return_value
    assert cmd_handler_instance.handle.await_count == 1
    # Cleanup path: db.close ran inside the with block's inner finally.
    db = main_mod.init_db.return_value
    assert db.close.await_count == 1
    # ChatLogger is now used as a context manager; __exit__ runs once when the
    # with block exits (whether cleanly or via exception). Production code's
    # __exit__ internally calls close_session, but on a MagicMock __exit__ does
    # not invoke close_session — so we assert __exit__ was triggered instead
    # (the public contract of the with statement).
    chat_logger_instance = main_mod.ChatLogger.return_value
    assert chat_logger_instance.__exit__.call_count == 1


@pytest.mark.asyncio
async def test_main_invokes_build_agent_without_db(monkeypatch):
    """Spec §11 / Task 13 + Phase 2: build_agent(default_llm_service) — db 已删除,
    Phase 2 改为接受单一 LlmService 参数。断言调用参数(而非仅断言 main() 返回)
    可以捕获"有人重新加回 db"或"绕开 LlmService DI"等回归。"""
    fake_build = AsyncMock(name="build_agent")
    _install_wiring(monkeypatch, fake_build)

    await main()

    fake_build.assert_awaited_once()
    args, kwargs = fake_build.await_args
    # Only LlmService positional; no positional db.
    assert len(args) == 1
    # Defensive: db must not slip in via keyword either.
    assert "db" not in kwargs
    assert "checkpointer" not in kwargs
