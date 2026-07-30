from pathlib import Path
from typing import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture(autouse=True)
async def _close_aiosqlite_connections():
    """Track every aiosqlite connection opened during a test and close
    them on teardown.

    Without this, aiosqlite worker threads block on their tx queue
    forever, holding a reference to the event loop. pytest-asyncio
    closes the loop but the process won't exit (thread still alive),
    which means anything buffering stdout until exit (RTK hook, or a
    simple `pytest | tail` waiting on pipe EOF) appears to hang.
    """
    import aiosqlite

    original_connect = aiosqlite.connect
    opened: list = []

    async def _tracking(*args, **kwargs):
        conn = await original_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    aiosqlite.connect = _tracking
    try:
        yield
    finally:
        aiosqlite.connect = original_connect
        for conn in opened:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001, S110
                pass


@pytest.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[AsyncClient, None]:
    db_file = str(tmp_path / "test_integration.db")
    monkeypatch.setenv("DB_PATH", db_file)

    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
