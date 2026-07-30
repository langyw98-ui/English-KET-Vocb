"""app.py lifespan shutdown 异常路径测试。

被测单元: src/api/app.py 中 lifespan 的 finally 块。
覆盖 §1.1 要求: shutdown except 元组必须只含外部失败类型,
不吞 ValueError/AttributeError 等代码 bug。
"""
from unittest.mock import AsyncMock, MagicMock

import aiosqlite

from src.api import app as app_module


def _patch_startup(monkeypatch, db: object, agent_inner: object) -> None:
    """短路 lifespan startup 的重 IO, 仅保留被测 finally 路径。

    lifespan 顺序: init_db -> AsyncSqliteSaver(db).setup -> build_agent。
    把这三项替换为直接返回注入对象的 stub。
    """
    monkeypatch.setattr(app_module, "init_db", AsyncMock(return_value=db))
    monkeypatch.setattr(
        app_module.AsyncSqliteSaver, "setup", AsyncMock(return_value=None)
    )
    fake_agent = MagicMock()
    fake_agent.agent = agent_inner
    monkeypatch.setattr(app_module, "build_agent", AsyncMock(return_value=fake_agent))


async def _drive_lifespan_and_shutdown(fake_app: MagicMock) -> None:
    """进入 lifespan 并立即退出, 触发 finally 块。"""
    async with app_module.lifespan(fake_app):
        # lifespan 在 yield 前已把 db/agent 挂到 app.state;
        # 不执行任何请求, 直接退出以触发 finally。
        pass


async def test_lifespan_db_close_failure_logs_warning(monkeypatch) -> None:
    """db.close() 抛 aiosqlite.Error 时, finally 记录 warning 而不崩溃。"""
    captured: list[str] = []

    def fake_warning(msg: str, *args: object, **kwargs: object) -> None:
        captured.append(msg)

    monkeypatch.setattr(app_module.logger, "warning", fake_warning)

    fake_db = MagicMock()
    fake_db.close = AsyncMock(side_effect=aiosqlite.Error("simulated close failure"))

    inner = MagicMock()
    inner.aclose = AsyncMock()
    _patch_startup(monkeypatch, fake_db, inner)

    fake_app = MagicMock()
    await _drive_lifespan_and_shutdown(fake_app)

    assert any("db.close" in m for m in captured), f"期望记录 db.close 失败, 实际: {captured}"
    fake_db.close.assert_awaited_once()


async def test_lifespan_agent_aclose_failure_logs_warning(monkeypatch) -> None:
    """agent.aclose() 抛 RuntimeError 时, finally 记录 warning 而不崩溃。"""
    captured: list[str] = []

    def fake_warning(msg: str, *args: object, **kwargs: object) -> None:
        captured.append(msg)

    monkeypatch.setattr(app_module.logger, "warning", fake_warning)

    fake_db = MagicMock()
    fake_db.close = AsyncMock()

    inner = MagicMock()
    inner.aclose = AsyncMock(side_effect=RuntimeError("agent shutdown boom"))
    _patch_startup(monkeypatch, fake_db, inner)

    fake_app = MagicMock()
    await _drive_lifespan_and_shutdown(fake_app)

    assert any("agent.aclose" in m for m in captured), (
        f"期望记录 agent.aclose 失败, 实际: {captured}"
    )
    inner.aclose.assert_awaited_once()


async def test_lifespan_does_not_swallow_value_error_code_bugs(monkeypatch) -> None:
    """db.close() 抛 ValueError(代码 bug)时, finally 必须让其上抛, 不静默兜底。

    验证 §1.1 第 5 条: except 元组不含 ValueError 等通用异常。
    """
    monkeypatch.setattr(app_module.logger, "warning", lambda *a, **k: None)

    fake_db = MagicMock()
    fake_db.close = AsyncMock(side_effect=ValueError("code bug"))

    inner = MagicMock()
    inner.aclose = AsyncMock()
    _patch_startup(monkeypatch, fake_db, inner)

    fake_app = MagicMock()
    try:
        await _drive_lifespan_and_shutdown(fake_app)
    except ValueError:
        return
    raise AssertionError("ValueError 应上抛而非被 finally 静默吞掉")


async def test_lifespan_does_not_swallow_attribute_error_code_bugs(monkeypatch) -> None:
    """agent.aclose() 抛 AttributeError(代码 bug)时, finally 必须让其上抛。"""
    monkeypatch.setattr(app_module.logger, "warning", lambda *a, **k: None)

    fake_db = MagicMock()
    fake_db.close = AsyncMock()

    inner = MagicMock()
    inner.aclose = AsyncMock(side_effect=AttributeError("code bug"))
    _patch_startup(monkeypatch, fake_db, inner)

    fake_app = MagicMock()
    try:
        await _drive_lifespan_and_shutdown(fake_app)
    except AttributeError:
        return
    raise AssertionError("AttributeError 应上抛而非被 finally 静默吞掉")


def test_shutdown_exception_tuples_exclude_code_bugs() -> None:
    """源码检查: shutdown 路径不得再含 'except Exception' 裸捕获。

    这是回归安全网: 即使 lifespan 结构后续被重构, 此断言仍能
    捕获裸 except Exception 的重新引入。
    """
    import inspect

    src = inspect.getsource(app_module)
    assert "except Exception" not in src, (
        "app.py 仍含 'except Exception', 违反 CLAUDE.md §1.1"
    )
    # 同时确认具体外部失败类型确实出现在源码中
    assert "asyncio.TimeoutError" in src
    assert "aiosqlite.Error" in src
    assert "OSError" in src
